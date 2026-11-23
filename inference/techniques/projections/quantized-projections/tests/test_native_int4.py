"""quantized-projections' AWQ and GPTQ matrices (crit's INT4 rows with a zero point a group) against an FP64 reference
of a small synthetic checkpoint's values, on the tile, decode-gemv's INT4 GEMV and the torch fallbacks, GPTQ's act order
included, each decode row the same bits alone and batched and every prompt row its decode row's on the tile; and the
native prompt GEMMs (prompt_gemm native: fp8-block-rows' and int-block-experts' gemm-native) against the declared
arithmetic's FP64 product and the provider path within the structures' tolerance, their facts saying the prompt rows
are not row invariant."""
import unittest
from unittest.mock import patch

from crit_plugins.testing import PluginTestCase

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def pack_words(torch, v, order):
    """[..., 8 j + e] values 0..15 -> int32 [..., j], nibble p holding element order[p]."""
    v = v.reshape(*v.shape[:-1], -1, 8).long()
    word = torch.zeros(v.shape[:-1], dtype=torch.int64)
    for p in range(8):
        word |= (v[..., order[p]] & 15) << (4 * p)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


class Int4Checkpoint:
    """A synthetic matrix as AWQ and GPTQ store it: codes [K][N] 0..15, zero points [K / group][N] 1..15, FP16
    scales, g_idx (act order: a permutation of the plain groups); the FP64 values [N][K]."""

    def __init__(self, torch, n, k, group, seed, act_order=False):
        g = torch.Generator().manual_seed(seed)
        self.q = torch.randint(0, 16, (k, n), generator=g)
        self.z = torch.randint(1, 16, (k // group, n), generator=g)
        self.s = ((torch.rand(k // group, n, generator=g) + 0.5) / 256).half()
        self.g_idx = torch.arange(k) // group
        if act_order:
            self.g_idx = self.g_idx[torch.randperm(k, generator=g)]
        self.values = ((self.q.double() - self.z[self.g_idx].double()) * self.s[self.g_idx].double()).t()
        self.torch = torch

    def awq(self):
        t = self.torch
        return self.cuda({"qweight": pack_words(t, self.q, AWQ_ORDER), "qzeros": pack_words(t, self.z, AWQ_ORDER),
                          "scales": self.s})

    def cuda(self, tensors):
        return {k: v.cuda() if isinstance(v, self.torch.Tensor) else v for k, v in tensors.items()}

    def gptq(self, zero_offset=1):
        t, seq = self.torch, list(range(8))
        found = {"qweight": pack_words(t, self.q.t(), seq).t().contiguous(),
                 "qzeros": pack_words(t, self.z - zero_offset, seq), "scales": self.s, "g_idx": self.g_idx}
        if zero_offset != 1:
            found["zero_offset"] = zero_offset
        return self.cuda(found)


class NativeAndInt4Tests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("these matrices multiply on CUDA")
        import importlib
        import torch
        from crit_plugins.assemble import load_module
        self.torch = torch
        self.dm = load_module(self.registry()["quantized-projections"]).device_matrices
        self.nv = importlib.import_module(self.dm.__package__ + ".device_nvfp4")

    def x(self, rows, k, seed=8):
        return self.torch.randn(rows, k, generator=self.torch.Generator().manual_seed(seed)).cuda().to(
            self.torch.bfloat16)

    def within(self, got, x, values, bound, what):
        exact = x.double() @ values.t().cuda()
        magnitude = x.double().abs() @ values.abs().t().cuda()
        excess = float(((got.double() - exact).abs() - magnitude * bound).max())
        self.assertLessEqual(excess, 1e-30, what)

    def test_awq_and_gptq_matrices_multiply_their_exact_values_and_keep_their_rows(self):
        torch = self.torch
        n, k = 384, 1024
        for group, act_order in ((128, False), (64, True)):
            ck = Int4Checkpoint(torch, n, k, group, seed=group, act_order=act_order)
            x = self.x(200, k)
            for gemv in ("auto", "native", "crossover"):
                plugin = self.assemble(options={"gemv": gemv})["projections"]
                stored = [("gptq", ck.gptq()), ("gptq", ck.gptq(0))] + ([] if act_order else [("awq", ck.awq())])
                for format, tensors in stored:
                    m = plugin.load(tensors, format)
                    what = (gemv, format, group, act_order)
                    self.assertEqual((m.format, m.rows, m.columns, m.group), ("int4_zp", n, k, group), what)
                    self.assertEqual(m.perm is not None, act_order, what)
                    full = plugin.matmul(x, m, torch.float32)
                    self.within(full, x, ck.values, 2e-5, what)
                    alone = [plugin.matmul(x[r:r + 1].clone(), m, torch.float32)[0] for r in range(16)]
                    for rows in (2, 7, 16):
                        batch = plugin.matmul(x[:rows], m, torch.float32)
                        self.within(batch, x[:rows], ck.values, 2e-5, what + (rows,))
                        for r in range(rows):
                            self.assertTrue(torch.equal(batch[r], alone[r]), what + (rows, r))
                    if m.mma is not None:   # every prompt row its decode row's bits (the prompt tile)
                        for r in (0, 15, 120, 199):
                            self.assertTrue(torch.equal(full[r], plugin.matmul(x[r:r + 1].clone(), m,
                                                                               torch.float32)[0]), what + (r,))
                    self.assertEqual(plugin.matmul(x[:3], m).dtype, torch.bfloat16)
                facts = plugin.facts()
                self.assertEqual(facts["prompt_arithmetic"]["int4_zp"],
                                 {"weights": "exact", "activations": "bf16", "sums": "fp32"})
                self.assertEqual(facts["prompt_kernels"]["int4_zp"], ["mma-rows prompt-tile"])
                self.assertTrue(facts["row_invariance"]["prompt_rows"])

    def test_the_torch_repack_and_fallbacks_give_the_kernels_bytes_and_values(self):
        torch = self.torch
        ck = Int4Checkpoint(torch, 256, 512, 64, seed=3, act_order=True)
        dev = torch.device("cuda")
        for format, tensors in (("gptq", ck.gptq()), ("gptq", ck.gptq(0)), ("awq", Int4Checkpoint(
                torch, 256, 512, 128, seed=4).awq())):
            kernel = self.dm.repack_int4(tensors, format, dev)
            with patch.object(self.nv, "rows_kernel", lambda: None):
                plain = self.dm.repack_int4(tensors, format, dev)
            for a, b in zip(kernel, plain):
                self.assertTrue(a == b if not isinstance(a, torch.Tensor) else torch.equal(a, b), format)
        x = self.x(40, 512)
        with patch.object(self.nv, "kernel", lambda name: None):   # no crit kernel: the torch fallbacks
            plugin = self.assemble()["projections"]
            m = plugin.load(ck.gptq(), "gptq")
            self.assertEqual((m.path, m.prompt_kernel), ("torch", "torch"))
            self.within(plugin.matmul(x, m, torch.float32), x, ck.values, 2e-5, "prompt")
            self.within(plugin.matmul(x[:5], m, torch.float32), x[:5], ck.values, 2e-5, "decode")

    def test_unreadable_int4_matrices_are_refused(self):
        torch = self.torch
        plugin = self.assemble()["projections"]
        ck = Int4Checkpoint(torch, 64, 256, 64, seed=1)
        bad = dict(ck.awq(), scales=ck.s[:, :32])
        with self.assertRaises(ValueError):
            plugin.load(bad, "awq")
        broken = dict(ck.gptq(), g_idx=torch.zeros(256, dtype=torch.int64))
        with self.assertRaises(ValueError):
            plugin.load(broken, "gptq")

    # ------------------------------------------------------------------------------------- native prompt GEMMs

    def fp8_channel(self, n, k, seed=2):
        torch = self.torch
        w = torch.randn(n, k, generator=torch.Generator().manual_seed(seed)) * 0.05
        scale = w.abs().amax(1) / 448
        return {"weight": (w / scale[:, None]).to(torch.float8_e4m3fn).cuda(), "scale": scale.cuda()}

    def mxfp8(self, n, k, seed=3):
        torch = self.torch
        blocks = (torch.randn(n, k, generator=torch.Generator().manual_seed(seed)) * 0.05).view(n, -1, 32)
        exponent = torch.ceil(torch.log2(blocks.abs().amax(-1) / 448)).clamp(-127, 127)
        codes = (blocks / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn).view(n, k)
        return {"weight": codes.cuda(), "scale": (exponent + 127).to(torch.uint8).cuda()}

    def nvfp4(self, n, k, runs=1, seed=4):
        torch = self.torch
        w = torch.randn(n, k, generator=torch.Generator().manual_seed(seed)).cuda() * 0.05
        g = torch.cat([torch.full((n // runs,), 448.0 * 6.0 / float(part.abs().max()))
                       for part in w.split(n // runs)]).cuda()
        codes, scales = [], []
        for part, gs in zip(w.split(n // runs), g.split(n // runs)):
            c, s = self.dm.nvfp4_rows(part, float(gs[0]))
            codes.append(c.view(torch.uint8))
            scales.append(self.dm.from_blocked(s, part.shape[0], k // 16).view(torch.uint8))
        return {"weight": torch.cat(codes), "scale": torch.cat(scales), "global_scale": g if runs > 1 else g[:1],
                "input_scale": torch.tensor(448.0 * 6.0 / 4.0)}

    def declared(self, x, m):
        """The FP64 product of the native path's quantized activations and the matrix's stored values."""
        values = self.dm.dense(m).double()
        if m.format.startswith("fp8"):
            q, scale = self.dm.fp8_rows(x)
            a = q.double() * scale.double()[:, None]
        elif m.format == "mxfp8":
            a = self.dm.mx_values(x).double()
        else:
            g = m.tensors["input_scale"]   # the decode rows' and the provider path's NVFP4 activations
            a = self.dm.nvfp4_values(x, g).double() / g
        return a @ values.t(), a.abs() @ values.abs().t()

    def test_native_prompt_rows_match_the_declared_arithmetic_and_the_provider_path(self):
        torch = self.torch
        if self.nv.fp8_block("cuda") is None or self.nv.int_block("cuda") is None:
            self.skipTest("fp8-block-rows' and int-block-experts' native GEMMs are not here")
        n, k = 640, 2048
        cases = (("fp8:channel", self.fp8_channel(n, k), {}, 2e-6, "fp8-block-rows gemm-native"),
                 ("mxfp8", self.mxfp8(n, k), {"mxfp8_activations": "mxfp8"}, 1e-5, "fp8-block-rows gemm-native"),
                 ("nvfp4", self.nvfp4(n, k), {}, 2e-6, "int-block-experts gemm-native"),
                 ("nvfp4", self.nvfp4(n, k, runs=5), {}, 2e-6, "int-block-experts gemm-native"))
        for format, tensors, options, bound, kernel in cases:
            native = self.assemble(options={**options, "prompt_gemm": "native"})["projections"]
            provider = self.assemble(options={**options, "prompt_gemm": "provider"})["projections"]
            m, p = native.load(tensors, format), provider.load(tensors, format)
            what = (format, options, m.runs if format == "nvfp4" else None)
            self.assertEqual(m.prompt_kernel, kernel, what)
            facts = native.facts()
            self.assertFalse(facts["row_invariance"]["prompt_rows"], what)
            self.assertEqual(facts["prompt_kernels"][format.split(":")[0]], [kernel], what)
            for rows in (17, 100, 384):
                x = self.x(rows, k, seed=rows)
                got = native.matmul(x, m, torch.float32).double()
                exact, magnitude = self.declared(x, m)
                self.assertLessEqual(float(((got - exact).abs() - magnitude * bound).max()), 1e-30, what + (rows,))
                # the provider path and the decode rows: the same arithmetic in other orders
                want = provider.matmul(x, p, torch.float32).double()
                self.assertLess(float((got - want).abs().max() / want.abs().max()), 1e-5, what + (rows,))
                self.assertEqual(native.matmul(x, m).dtype, torch.bfloat16)
            if format == "mxfp8":   # decode rows on the MXFP8 values: the prompt rows' arithmetic in another order
                x = self.x(9, k)
                exact, magnitude = self.declared(x, m)
                got = native.matmul(x, m, torch.float32).double()
                self.assertLessEqual(float(((got - exact).abs() - magnitude * 2e-5).max()), 1e-30, what)
                self.assertEqual(native.facts()["decode_activations"]["mxfp8"], "mxfp8")

    def test_prompt_gemm_auto_takes_the_native_gemm_where_it_measured_faster(self):
        if self.nv.fp8_block("cuda") is None:
            self.skipTest("fp8-block-rows is not here")
        plugin = self.assemble(options={"mxfp8_activations": "mxfp8"})["projections"]   # prompt_gemm auto
        found = {f: plugin.load(t, f).prompt_kernel for f, t in (("mxfp8", self.mxfp8(256, 1024)),
                                                                   ("fp8:channel", self.fp8_channel(256, 1024)),
                                                                   ("nvfp4", self.nvfp4(256, 1024)))}
        expected = {"mxfp8": "fp8-block-rows gemm-native",
                    "fp8:channel": "fp8-block-rows gemm-native" if self.dm.NATIVE_AUTO["fp8_rows"] else "provider",
                    "nvfp4": "int-block-experts gemm-native" if self.dm.NATIVE_AUTO["nvfp4"] else "provider"}
        self.assertEqual(found, expected)
        self.assertFalse(plugin.facts()["row_invariance"]["prompt_rows"])

    def test_native_runs_16_bit_prompts_on_fixed_tiles_and_auto_follows_its_kinds(self):
        torch = self.torch
        if None in (self.nv.fp8_block("cuda"), self.nv.int_block("cuda"), self.nv.tiles_kernel()):
            self.skipTest("fp8-block-rows, int-block-experts and fixed-tile-gemm are not all here")
        from unittest.mock import patch
        bf16 = {"weight": (torch.randn(384, 1024, generator=torch.Generator().manual_seed(5)) * 0.02).to(
            torch.bfloat16).cuda()}
        cases = (("bf16", bf16, {}), ("fp8:channel", self.fp8_channel(256, 1024), {"fp8_activations": "bf16"}),
                 ("mxfp8", self.mxfp8(256, 1024), {}), ("nvfp4", self.nvfp4(256, 1024), {"activations": "bf16"}))
        for format, tensors, options in cases:
            native = self.assemble(options={**options, "prompt_gemm": "native"})["projections"]
            provider = self.assemble(options={**options, "prompt_gemm": "provider"})["projections"]
            m, p = native.load(tensors, format), provider.load(tensors, format)
            self.assertEqual((m.prompt_kernel, p.prompt_kernel), ("fixed-tile-gemm", "provider"), format)
            x = self.x(300, 1024)
            got, want = native.matmul(x, m, torch.float32), provider.matmul(x, p, torch.float32)
            self.assertLess(float((got - want).abs().max() / want.abs().max()), 2e-5, format)
        # weight-only NVFP4 prompts in fp8: FP8-decoded chunks on fp8-block-rows' per-channel GEMM
        options = {"activations": "bf16", "prompt": "fp8"}
        native = self.assemble(options={**options, "prompt_gemm": "native"})["projections"]
        provider = self.assemble(options={**options, "prompt_gemm": "provider"})["projections"]
        tensors = self.nvfp4(640, 2048)
        m, p = native.load(tensors, "nvfp4"), provider.load(tensors, "nvfp4")
        self.assertEqual((m.prompt_kernel, p.prompt_kernel), ("fp8-block-rows gemm-native-channel", "provider"))
        x = self.x(300, 2048)
        got, want = native.matmul(x, m, torch.float32), provider.matmul(x, p, torch.float32)
        self.assertLess(float((got - want).abs().max() / want.abs().max()), 2e-5)
        self.assertEqual(native.matmul(x, m).dtype, torch.bfloat16)
        # every kind of NATIVE_AUTO True: auto is native everywhere
        with patch.dict(self.dm.NATIVE_AUTO, {kind: True for kind in self.dm.NATIVE_AUTO}):
            plugin = self.assemble(options={"mxfp8_activations": "mxfp8"})["projections"]
            found = {f: plugin.load(t, f).prompt_kernel for f, t in (("mxfp8", self.mxfp8(256, 1024)),
                                                                       ("fp8:channel", self.fp8_channel(256, 1024)),
                                                                       ("nvfp4", self.nvfp4(256, 1024)),
                                                                       ("bf16", bf16))}
        self.assertEqual(found, {"mxfp8": "fp8-block-rows gemm-native", "fp8:channel": "fp8-block-rows gemm-native",
                                 "nvfp4": "int-block-experts gemm-native", "bf16": "fixed-tile-gemm"})

    def test_the_mxfp8_quantizer_in_torch_gives_the_kernels_codes(self):
        torch = self.torch
        fb = self.nv.fp8_block("cuda")
        if fb is None:
            self.skipTest("fp8-block-rows is not here")
        x = self.x(64, 1024) * torch.logspace(-30, 30, 1024, base=2.0, device="cuda").to(torch.bfloat16)
        x[3, :64] = 0
        codes, scales = fb.quantize(x.contiguous(), 32, fb.MX)
        mine, theirs = self.dm.mx_codes(x)
        self.assertTrue(torch.equal(mine.view(torch.uint8), codes), "codes")
        self.assertTrue(torch.equal(theirs, scales), "scales")
        self.assertTrue(torch.equal(self.dm.mx_values(x, fb), self.dm.mx_values(x)))

    def test_native_prompt_rows_capture_and_replay(self):
        torch = self.torch
        if self.nv.fp8_block("cuda") is None or self.nv.int_block("cuda") is None:
            self.skipTest("the native GEMMs are not here")
        plugin = self.assemble(options={"prompt_gemm": "native"})["projections"]
        for format, tensors in (("fp8:channel", self.fp8_channel(256, 1024)), ("nvfp4", self.nvfp4(256, 1024))):
            m = plugin.load(tensors, format)
            x = self.x(96, 1024)
            eager = plugin.matmul(x, m)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = plugin.matmul(x, m)
            x.copy_(self.x(96, 1024, seed=99))
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out, plugin.matmul(x, m)), format)
            self.assertFalse(torch.equal(out, eager), format)


if __name__ == "__main__":
    unittest.main()
