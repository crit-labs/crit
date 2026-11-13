"""quantized-projections' device form: stored BF16, FP8 and NVFP4 matrices multiply as their decoded values do, the
NVFP4 codes and scales round trip, and each row's result is the same alone and in a batch, on crit's kernels' paths
(built on this GPU, with the engine's tuning or the default knobs) and on the torch fallbacks; draft-only re-encoding
gives the FP64 codec's codes."""
import struct
import unittest

from crit_plugins.testing import PluginTestCase


def torch_rows(torch, rows, width=256, seed=21):
    return torch.randn(rows, width, generator=torch.Generator().manual_seed(seed)).cuda().to(torch.bfloat16)


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class DeviceMatrixTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None
        if self.device() != "cuda":
            self.skipTest("FP8 and NVFP4 matrices multiply on CUDA (cuBLASLt)")
        self.plugin = self.assemble()["projections"]
        g = self.torch.Generator().manual_seed(5)
        self.w = self.torch.randn(96, 256, generator=g).cuda() * 0.1
        self.x = self.torch.randn(7, 256, generator=g).cuda()

    def nvfp4_weight(self):
        """The weight as an NVFP4 checkpoint stores it (the device codec's own quantization)."""
        from crit_plugins.assemble import load_module
        codec = load_module(self.registry()["quantized-projections"]).device_matrices
        global_scale = 448.0 * 6.0 / float(self.w.abs().max())
        codes, scales = codec.nvfp4_rows(self.w, global_scale)
        linear = codec.from_blocked(scales, 96, 16)
        return codec, {"weight": codes.view(self.torch.uint8), "scale": linear.view(self.torch.uint8),
                       "global_scale": self.torch.tensor(global_scale), "input_scale": self.torch.tensor(448.0 * 6.0 / 4.0)}

    def mxfp8_weight(self):
        """The weight as an MXFP8 checkpoint stores it: E4M3 codes and one E8M0 exponent per 32 inputs (the block's
        amax over 448, rounded up to a power of two)."""
        torch = self.torch
        blocks = self.w.view(96, -1, 32)
        exponent = torch.ceil(torch.log2(blocks.abs().amax(-1) / 448)).clamp(-127, 127)
        codes = (blocks / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn).view(96, 256)
        return {"weight": codes, "scale": (exponent + 127).to(torch.uint8)}

    def test_bf16_and_fp8_matrices_multiply_as_their_values(self):
        torch = self.torch
        bf16 = self.plugin.load({"weight": self.w}, "bf16")
        self.assertClose(self.plugin.matmul(self.x, bf16).float().flatten().tolist(),
                         (self.x @ self.w.t()).flatten().tolist(), atol=0.05, rtol=0.02)
        scale = self.w.abs().amax(dim=1) / 448
        stored = (self.w / scale[:, None]).to(torch.float8_e4m3fn)
        x = torch.randn(40, 256, generator=torch.Generator().manual_seed(2)).cuda()
        exact = x.to(torch.bfloat16).double() @ (stored.double() * scale.double()[:, None]).t()
        for decode in ("prompt", "bf16"):
            plugin = self.assemble(options={"decode_activations": decode})["projections"]
            fp8 = plugin.load({"weight": stored, "scale": scale}, "fp8:channel")
            # Decode rows (up to 16): the prompt rows' E4M3 activations (at most a relative step of 1/16 per product),
            # or W8A16 (exact products in FP32 sums). Prompt rows: the E4M3 activations through cuBLASLt.
            prompt = plugin.matmul(x, fp8, torch.float32).double()
            for rows, bound in ((7, 1e-5 if decode == "bf16" else 0.06), (16, 1e-5 if decode == "bf16" else 0.06),
                                (40, 0.06)):
                found = plugin.matmul(x[:rows], fp8, torch.float32).double()
                self.assertLess(float((found - exact[:rows]).abs().max() / exact[:rows].abs().max()), bound,
                                (decode, rows))
                if decode == "prompt":   # the same quantized activations: decode and prompt rows differ in order only
                    self.assertLess(float((found - prompt[:rows]).abs().max() / prompt[:rows].abs().max()), 1e-5, rows)
            tensor = plugin.load({"weight": stored, "scale": scale.max()}, "fp8:tensor")
            self.assertEqual(tuple(plugin.matmul(self.x, tensor).shape), (7, 96))
            self.assertEqual(plugin.facts()["decode_activations"], {"fp8": "fp8" if decode == "prompt" else "bf16"})
            self.assertIn(plugin.facts()["decode_paths"]["fp8"][0], ("mma", "dot", "gemv", "cublaslt", "torch"))

    def test_mxfp8_matrices_multiply_as_their_exact_values_in_decode_and_prompt_rows(self):
        torch = self.torch
        stored = self.mxfp8_weight()
        plugin = self.assemble()["projections"]
        m = plugin.load(stored, "mxfp8")
        from crit_plugins.assemble import load_module
        values = load_module(self.registry()["quantized-projections"]).device_matrices.dense(m).double()
        self.assertLess(float((values - self.w.double()).abs().max()), float(self.w.abs().max()) / 16)
        x = torch.randn(40, 256, generator=torch.Generator().manual_seed(3)).cuda().to(torch.bfloat16)
        exact = x.double() @ values.t()
        for rows in (1, 9, 16, 40):   # decode rows (W8A16 GEMV) and prompt rows (exact BF16 values, torch's GEMM)
            found = plugin.matmul(x[:rows], m, torch.float32).double()
            self.assertLess(float((found - exact[:rows]).abs().max() / exact[:rows].abs().max()), 1e-5, rows)
        self.assertEqual(plugin.facts()["prompt_activations"], {"mxfp8": "bf16"})
        with self.assertRaisesRegex(ValueError, "E8M0 scales"):
            plugin.load({**stored, "scale": stored["scale"][:, :4]}, "mxfp8")

    def test_w4a4_decode_rows_take_the_prompt_rows_activations_unless_decode_bf16(self):
        torch = self.torch
        codec, stored = self.nvfp4_weight()
        x = torch.randn(40, 256, generator=torch.Generator().manual_seed(4)).cuda().to(torch.bfloat16)
        found = {}
        for decode in ("prompt", "bf16"):
            plugin = self.assemble(options={"decode_activations": decode})["projections"]
            m = plugin.load(stored, "nvfp4")
            self.assertNotIn("scale_rows", m.tensors)   # the decode GEMV reads the blocked scales: no second copy
            values = codec.dense(m).double()
            weight_only = x.double() @ values.t()
            found[decode] = [plugin.matmul(x[:rows], m, torch.float32).double() for rows in (1, 16, 40)]
            self.assertEqual(plugin.facts()["decode_activations"], {"nvfp4": "nvfp4" if decode == "prompt" else "bf16"})
            self.assertEqual(plugin.facts()["prompt_activations"], {"nvfp4": "stored"})
        prompt = found["prompt"][2]
        for rows, a, b in zip((1, 16), found["prompt"], found["bf16"]):
            # prompt: the same NVFP4 activations as the prompt rows, the sums in another order; bf16: weight only
            self.assertLess(float((a - prompt[:rows]).abs().max() / prompt[:rows].abs().max()), 1e-5, rows)
            self.assertLess(float((b - weight_only[:rows]).abs().max() / weight_only[:rows].abs().max()), 1e-5, rows)
            self.assertGreater(float((a - weight_only[:rows]).abs().max() / weight_only[:rows].abs().max()), 1e-4)
        self.assertTrue(torch.equal(found["prompt"][2], found["bf16"][2]))   # prompt rows: W4A4 either way

    def test_nvfp4_codes_and_scales_round_trip_and_multiply_as_their_values(self):
        codec, stored = self.nvfp4_weight()
        m = self.plugin.load(stored, "nvfp4")
        values = codec.dense(m)
        self.assertLess(float((values - self.w).abs().max()), float(self.w.abs().max()) / 6 + 1e-6)
        # Prompt rows: the activations as matmul encodes them, decoded: the product of the two decoded operands.
        x = self.torch.randn(20, 256, generator=self.torch.Generator().manual_seed(7)).cuda()
        codes, scales = codec.nvfp4_rows(x, m.tensors["input_scale"])
        decoded = (codec.decode_e2m1(codes.view(self.torch.uint8)).view(20, -1, 16)
                   * codec.from_blocked(scales, 20, 16).float()[..., None]).reshape(20, 256) / m.tensors["input_scale"]
        exact = decoded @ values.t()
        found = self.plugin.matmul(x, m).float()
        self.assertLess(float((found - exact).abs().max() / exact.abs().max()), 0.01)

    def test_a_row_keeps_its_bits_alone_and_among_up_to_16_rows_in_every_format(self):
        torch = self.torch
        codec, stored = self.nvfp4_weight()
        scale = self.w.abs().amax(dim=1) / 448
        fp8 = (self.w / scale[:, None]).to(torch.float8_e4m3fn)
        weight_only = self.assemble(options={"activations": "bf16"})["projections"]
        stored_rows = self.assemble(options={"decode_activations": "bf16"})["projections"]
        matrices = [(self.plugin, self.plugin.load({"weight": self.w}, "bf16")),
                    (self.plugin, self.plugin.load({"weight": fp8, "scale": scale}, "fp8:channel")),
                    (self.plugin, self.plugin.load({"weight": fp8, "scale": scale.max()}, "fp8:tensor")),
                    (self.plugin, self.plugin.load(self.mxfp8_weight(), "mxfp8")),
                    (self.plugin, self.plugin.load(stored, "nvfp4")),
                    (weight_only, weight_only.load(stored, "nvfp4")),
                    (stored_rows, stored_rows.load({"weight": fp8, "scale": scale}, "fp8:channel")),
                    (stored_rows, stored_rows.load(stored, "nvfp4"))]
        x = torch.randn(16, 256, generator=torch.Generator().manual_seed(12)).cuda().to(torch.bfloat16)
        for plugin, m in matrices:
            most = min(16, plugin.facts()["row_invariance"]["rows"])   # past 16: test_tiles
            self.assertIn(most, (8, 16))
            for out_dtype in (None, torch.float32):
                alone = [plugin.matmul(x[r:r + 1].clone(), m, out_dtype)[0] for r in range(most)]
                for rows in range(2, most + 1):
                    batch = plugin.matmul(x[:rows], m, out_dtype)
                    for r in range(rows):
                        self.assertTrue(torch.equal(batch[r], alone[r]),
                                        (m.format, m.activations, m.decode, rows, r, out_dtype))

    def test_fixed_tiles_keep_prompt_rows_bits_and_the_facts_say_so(self):
        torch = self.torch
        codec, stored = self.nvfp4_weight()
        plugin = self.assemble(options={"activations": "bf16", "prompt_gemm": "fixed_tiles"})["projections"]
        x = torch.randn(300, 256, generator=torch.Generator().manual_seed(13)).cuda().to(torch.bfloat16)
        for m in (plugin.load({"weight": self.w}, "bf16"), plugin.load(stored, "nvfp4")):
            whole = plugin.matmul(x, m)
            for first, rows in ((0, 17), (7, 40), (100, 200)):   # more rows than a decode call takes
                self.assertTrue(torch.equal(plugin.matmul(x[first:first + rows].clone(), m), whole[first:first + rows]),
                                (m.format, first, rows))
        self.assertEqual(plugin.facts()["row_invariance"], {"rows": 16, "prompt_rows": True,
                                                             "prompt_exact": False})
        # One matrix through cuBLAS for its prompt rows, and the facts no longer promise them.
        mixed = self.assemble(options={"activations": "bf16", "prompt_gemm": "fixed_tiles",
                                       "prompt_roles": "gate_up=fp8"})["projections"]
        mixed.load(stored, "nvfp4")
        self.assertTrue(mixed.facts()["row_invariance"]["prompt_rows"])
        mixed.load(stored, "nvfp4", role="gate_up")
        self.assertEqual(mixed.facts()["row_invariance"], {"rows": 16, "prompt_rows": False, "prompt_exact": False})

    def test_a_matrix_of_the_wrong_shape_is_refused(self):
        with self.assertRaisesRegex(ValueError, "one scale per output row"):
            self.plugin.load({"weight": self.w.to(self.torch.float8_e4m3fn), "scale": self.torch.ones(3)}, "fp8:channel")
        with self.assertRaisesRegex(ValueError, "multiples of 16, not 40 x 256"):   # at load, not at the first matmul
            self.plugin.load({"weight": self.w[:40].to(self.torch.float8_e4m3fn), "scale": self.torch.ones(40)},
                             "fp8:channel")
        with self.assertRaisesRegex(ValueError, "is bf16, fp8:channel, fp8:tensor, mxfp8, nvfp4"):
            self.plugin.load({"weight": self.w}, "int3")
        codec, stored = self.nvfp4_weight()
        with self.assertRaisesRegex(ValueError, "one global scale or one per row, not 3"):
            self.plugin.load({**stored, "global_scale": self.torch.ones(3)}, "nvfp4")
        with self.assertRaisesRegex(ValueError, "one input scale, not 2"):
            self.plugin.load({**stored, "input_scale": self.torch.ones(2)}, "nvfp4")

    def test_matrices_land_on_the_engine_device_whatever_device_their_tensors_are_on(self):
        class Context:
            device = "cuda"
        plugin = self.assemble(context=Context())["projections"]
        codec, stored = self.nvfp4_weight()
        for m in (plugin.load({"weight": self.w.cpu()}, "bf16"),
                  plugin.load({k: v.cpu() for k, v in stored.items()}, "nvfp4")):
            self.assertTrue(all(t.is_cuda for t in m.tensors.values() if hasattr(t, "is_cuda")), m.format)
            self.assertEqual(tuple(plugin.matmul(self.x, m).shape), (7, 96))

    def test_weight_only_nvfp4_multiplies_16_bit_activations_by_the_stored_values(self):
        # activations bf16: the GEMV (up to 8 rows) and the decoded GEMM (more) both give x times the stored values.
        plugin = self.assemble(options={"activations": "bf16"})["projections"]
        codec, stored = self.nvfp4_weight()
        m = plugin.load(stored, "nvfp4")
        values = codec.dense(m)
        x = self.torch.randn(20, 256, generator=self.torch.Generator().manual_seed(9)).cuda().to(self.torch.bfloat16)
        for rows in (1, 2, 5, 8, 9, 20):
            exact = x[:rows].float() @ values.t()
            found = plugin.matmul(x[:rows], m).float()
            self.assertLess(float((found - exact).abs().max() / exact.abs().max()), 1e-2, rows)
        # A decode row is the same alone and among up to 8 (the GEMV's order does not depend on the row count).
        batch = plugin.matmul(x[:8], m)
        for t in range(8):
            self.assertTrue(bool((plugin.matmul(x[t:t + 1], m) == batch[t:t + 1]).all()), t)

    def test_weight_only_nvfp4_prompt_rows_differ_from_decode_rows_in_summation_order_only(self):
        # The prompt rows multiply e2m1 x scale (exact in BF16) and apply the inverse global scale to the FP32 sums,
        # as the decode GEMV does: a row's FP32 result in a 17-row call is within FP32 rounding of the row alone.
        torch = self.torch
        codec, stored = self.nvfp4_weight()
        for options in ({"activations": "bf16"}, {"activations": "bf16", "prompt_gemm": "fixed_tiles"}):
            plugin = self.assemble(options=options)["projections"]
            m = plugin.load(stored, "nvfp4")
            x = torch.randn(17, 256, generator=torch.Generator().manual_seed(14)).cuda().to(torch.bfloat16)
            prompt = plugin.matmul(x, m, torch.float32)
            decode = torch.cat([plugin.matmul(x[r:r + 1], m, torch.float32) for r in range(17)])
            self.assertLess(float((prompt - decode).abs().max() / decode.abs().max()), 1e-5, options)

    def test_decode_rows_caps_the_rows_a_decode_call_takes_and_the_facts_promise(self):
        torch = self.torch
        codec, stored = self.nvfp4_weight()
        plugin = self.assemble(options={"decode_rows": 8, "gemv": "native"})["projections"]   # the tile: test_tiles
        m = plugin.load(stored, "nvfp4")
        self.assertEqual((m.decode_rows, plugin.facts()["row_invariance"]["rows"]), (8, 8))
        for weight_only in (plugin.load({"weight": self.w}, "bf16"),
                            self.assemble(options={"decode_rows": 12, "activations": "bf16"})["projections"].load(
                                stored, "nvfp4")):
            self.assertEqual(weight_only.decode_rows, 8 if weight_only.format == "bf16" else 12)
        x = torch.randn(9, 256, generator=torch.Generator().manual_seed(15)).cuda().to(torch.bfloat16)
        from crit_plugins.assemble import load_module
        matrices = load_module(self.registry()["quantized-projections"]).device_matrices
        self.assertTrue(torch.equal(plugin.matmul(x, m), matrices._prompt_rows(x, m, None)))   # 9 rows: prompt rows
        for rows in (7, 17):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "8 to 16 rows"):
                matrices.load(stored, "nvfp4", "cuda", decode_rows=rows)

    def test_fixed_tile_prompt_gemms_give_a_row_the_same_result_whatever_rows_share_them(self):
        plugin = self.assemble(options={"activations": "bf16", "prompt_gemm": "fixed_tiles"})["projections"]
        codec, stored = self.nvfp4_weight()
        m = plugin.load(stored, "nvfp4")
        values = codec.dense(m)
        torch = self.torch
        # 45,056 rows of a 96-row matrix fill 352 large tiles, the large one on a device of up to 176 multiprocessors;
        # the slices below fill too few and take the small one: the switch keeps every row's order.
        rows_all = 45056
        x = torch.randn(rows_all, 256, generator=torch.Generator().manual_seed(4)).cuda().to(torch.bfloat16)
        batch = plugin.matmul(x, m)
        exact = x.float() @ values.t()
        self.assertLess(float((batch.float() - exact).abs().max() / exact.abs().max()), 1e-2)
        for first, rows in ((0, 17), (17, 40), (300, 700), (rows_all - 17, 17)):
            self.assertTrue(torch.equal(plugin.matmul(x[first:first + rows], m), batch[first:first + rows]),
                            (first, rows))
        with self.assertRaises(ValueError):
            self.assemble(options={"activations": "bf16", "prompt": "fp8", "prompt_gemm": "fixed_tiles"})

    def test_prompt_roles_pick_a_matrix_s_prompt_arithmetic_by_its_role(self):
        options = {"activations": "bf16", "prompt": "fp16_acc", "prompt_roles": "gate_up=fp8, down=bf16"}
        plugin = self.assemble(options=options)["projections"]
        codec, stored = self.nvfp4_weight()
        torch = self.torch
        x = torch.randn(20, 256, generator=torch.Generator().manual_seed(6)).cuda().to(torch.bfloat16)
        exact = x.float() @ codec.dense(plugin.load(stored, "nvfp4")).t()
        for role, arith in (("gate_up", "fp8"), ("down", "bf16"), ("qkv", "fp16_acc"), (None, "fp16_acc")):
            m = plugin.load(stored, "nvfp4", role=role)
            self.assertEqual(m.prompt, arith, role)
            found = plugin.matmul(x, m).float()
            self.assertLess(float((found - exact).abs().max() / exact.abs().max()), 5e-2, role)
        self.assertEqual(plugin.facts()["prompt_activations"]["nvfp4_by_role"], {"gate_up": "fp8", "down": "bf16"})
        with self.assertRaises(ValueError):
            self.assemble(options={"activations": "bf16", "prompt_roles": "gate_up=fp4"})

    def native_module(self):
        import importlib
        from crit_plugins.assemble import load_module
        matrices = load_module(self.registry()["quantized-projections"]).device_matrices
        return importlib.import_module(matrices.__package__ + ".device_nvfp4")

    def paths(self):
        """The decode-row paths this machine has: crit's kernel decode-gemv's CUDA and Triton GEMVs where it builds
        here, and always the torch fallback."""
        device_nvfp4 = self.native_module()
        found = ["fallback"]
        if device_nvfp4.native("cuda") is not None:
            found += ["native", "triton"]
        return device_nvfp4, found

    def load_on(self, path, stored, options=None):
        """A weight-only NVFP4 matrix loaded on one decode-row path (the fallback: as if no crit kernel were here)."""
        device_nvfp4 = self.native_module()
        plugin = self.assemble(options={"activations": "bf16", "gemv": "triton" if path == "triton" else "native",
                                        **(options or {})})["projections"]
        if path != "fallback":
            return plugin, plugin.load(stored, "nvfp4")
        from unittest.mock import patch
        with patch.object(device_nvfp4, "kernel", lambda name: None):   # as if no crit kernel were here
            return plugin, plugin.load(stored, "nvfp4")

    def test_every_gemv_path_multiplies_by_the_stored_values_and_keeps_a_row_s_bits_alone_and_batched(self):
        torch = self.torch
        device_nvfp4, paths = self.paths()
        codec, stored = self.nvfp4_weight()
        x = torch.randn(16, 256, generator=torch.Generator().manual_seed(8)).cuda().to(torch.bfloat16)
        found = {}
        for path in paths:
            plugin, m = self.load_on(path, stored)
            self.assertEqual((m.native is not None, m.triton is not None), (path == "native", path == "triton"))
            self.assertEqual(m.decode_rows, 8 if path == "triton" else 16)
            exact = x[:8].float() @ codec.dense(m).t()
            for dtype in (None, torch.float32):
                batch = plugin.matmul(x[:8], m, dtype)
                for rows in (1, 3, 8, *((12, 16) if path != "triton" else ())):
                    for r in range(rows):
                        alone = plugin.matmul(x[r:r + 1].clone(), m, dtype)
                        self.assertTrue(torch.equal(alone[0], plugin.matmul(x[:rows], m, dtype)[r]), (path, rows, r))
                self.assertLess(float((batch.float() - exact).abs().max() / exact.abs().max()), 1e-2, path)
            found[path] = plugin.matmul(x[:8], m, torch.float32)
            # 16-bit FP16 activations, a misaligned view and an output of another dtype take the same path.
            wide = torch.zeros(3, 260, dtype=torch.bfloat16, device="cuda")   # rows 8 bytes off 16-byte boundaries
            wide[:, 4:] = x[:3]
            t = m.tensors
            self.assertTrue(torch.equal(device_nvfp4.gemv(wide[:, 4:], t["codes"], t["scale"], t["inverse"],
                                                          native=m.native, triton=m.triton),
                                        device_nvfp4.gemv(x[:3], t["codes"], t["scale"], t["inverse"],
                                                          native=m.native, triton=m.triton)), path)
            half = device_nvfp4.gemv(x[:2].half(), t["codes"], t["scale"], t["inverse"], dtype=torch.float16,
                                     native=m.native, triton=m.triton)
            self.assertEqual(half.dtype, torch.float16)
        # All in FP32, in other orders: within FP32 rounding of each other.
        for path in paths[1:]:
            self.assertLess(float((found[path] - found["fallback"]).abs().max() / found["fallback"].abs().max()),
                            1e-5, path)

    def test_the_fallback_decoders_give_the_kernel_s_bits(self):
        torch = self.torch
        device_nvfp4, paths = self.paths()
        g = torch.Generator().manual_seed(5)
        codes = torch.randint(0, 256, (96, 128), generator=g, dtype=torch.uint8).cuda()
        scales = (torch.rand(96, 16, generator=g) * 4 + 0.01).to(torch.float8_e4m3fn).cuda()
        inverse = (torch.rand(96, generator=g) + 0.5).cuda() / 64
        factor = inverse / device_nvfp4.row_scales(scales, inverse)
        values = device_nvfp4._values(codes, scales, None, torch.float64) * inverse.double()[:, None]
        for dtype in (torch.bfloat16, torch.float16):
            out = device_nvfp4.decode(codes, scales, inverse, torch.empty(96, 256, dtype=dtype, device="cuda"))
            # e2m1 * scale * inverse rounded once: within half a 16-bit step of the exact value
            self.assertTrue(bool(((out.double() - values).abs() <= values.abs() * 2.0 ** -8 * 1.01 + 1e-7).all()), dtype)
        from unittest.mock import patch
        found = {}
        for path in ("kernel", "fallback")[0 if "native" in paths else 1:]:
            with patch.object(device_nvfp4, "kernel", device_nvfp4.kernel if path == "kernel" else lambda name: None):
                found[path] = (device_nvfp4.decode(codes, scales, inverse,
                                                   torch.empty(96, 256, dtype=torch.bfloat16, device="cuda")),
                               device_nvfp4.decode_fp8(codes, scales, factor,
                                                       torch.empty(96, 256, dtype=torch.float8_e4m3fn, device="cuda")))
        if "kernel" in found:
            for a, b in zip(found["kernel"], found["fallback"]):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    def test_the_facts_name_the_load_kernels_and_where_they_came_from(self):
        from crit_plugins import kernels
        codec, stored = self.nvfp4_weight()
        plugin = self.assemble(options={"activations": "bf16", "prompt_gemm": "fixed_tiles"})["projections"]
        m = plugin.load(stored, "nvfp4")
        plugin.matmul(torch_rows(self.torch, 20), m)
        record = kernels.loaded()
        for name in ("decode-gemv", "fixed-tile-gemm"):
            self.assertIn(name, record)
            self.assertIn(record[name]["source"], ("tuned", "built", "fallback"))
            if record[name]["source"] != "fallback":   # built with the engine's tuning where it has one for the kernel
                tuning = kernels.tuning_file(name)
                self.assertEqual(record[name]["source"], "tuned" if tuning else "built", record[name])
                self.assertEqual(record[name].get("tuning"), str(tuning) if tuning else None)
            self.assertIn("quantized-projections", record[name]["plugins"])

    def test_weight_only_prompt_rows_run_in_the_chosen_arithmetic(self):
        # fp16_acc: FP16 values and sums (a relative step of 1/1024 per term); fp8: E4M3 weights and activation rows
        # (3 mantissa bits each, about 6% per product at worst); decode rows stay on the GEMV either way.
        codec, stored = self.nvfp4_weight()
        x = self.torch.randn(20, 256, generator=self.torch.Generator().manual_seed(9)).cuda().to(self.torch.bfloat16)
        for prompt, bound, dtype in (("fp16_acc", 1e-2, self.torch.float16), ("fp8", 0.08, self.torch.bfloat16)):
            plugin = self.assemble(options={"activations": "bf16", "prompt": prompt})["projections"]
            m = plugin.load(stored, "nvfp4")
            exact = x.float() @ codec.dense(m).t()
            found = plugin.matmul(x, m)
            self.assertEqual(found.dtype, dtype, prompt)
            self.assertLess(float((found.float() - exact).abs().max() / exact.abs().max()), bound, prompt)
            one = plugin.matmul(x[:1], m).float()
            self.assertLess(float((one - exact[:1]).abs().max() / exact[:1].abs().max()), 1e-2, prompt)

    def test_it_reports_the_prompt_activations_of_what_it_loaded(self):
        codec, stored = self.nvfp4_weight()
        assembly = self.assemble(options={"activations": "bf16", "prompt": "fp16_acc"})
        plugin = assembly["projections"]
        self.assertEqual(plugin.facts(),
                         {"row_invariance": {"rows": 16, "prompt_rows": False, "prompt_exact": False}})
        plugin.load(stored, "nvfp4")
        self.assertEqual(plugin.facts()["prompt_activations"], {"nvfp4": "fp16_acc"})
        self.assertEqual(assembly.facts()["prompt_activations"], {"nvfp4": "fp16_acc"})
        w4a4 = self.assemble()["projections"]
        w4a4.load(stored, "nvfp4")
        self.assertEqual(w4a4.facts()["prompt_activations"], {"nvfp4": "stored"})

    def test_matrices_stacked_by_rows_keep_each_ones_global_scale(self):
        codec, first = self.nvfp4_weight()
        second = {**first, "global_scale": first["global_scale"] * 4}
        stacked = {"weight": self.torch.cat([first["weight"], second["weight"]]),
                   "scale": self.torch.cat([first["scale"], second["scale"]]),
                   "global_scale": self.torch.cat([first["global_scale"].expand(96), second["global_scale"].expand(96)]),
                   "input_scale": first["input_scale"]}
        for activations in ("nvfp4", "bf16"):
            plugin = self.assemble(options={"activations": activations})["projections"]
            x = self.x.to(self.torch.bfloat16)
            together = plugin.matmul(x, plugin.load(stacked, "nvfp4")).float()
            apart = self.torch.cat([plugin.matmul(x, plugin.load(first, "nvfp4")),
                                    plugin.matmul(x, plugin.load(second, "nvfp4"))], dim=1).float()
            self.assertLess(float((together - apart).abs().max()), float(apart.abs().max()) * 1e-2, activations)

    def test_a_projection_captures_in_a_cuda_graph(self):
        # No host copy per call: a step multiplying through the plugin replays from a graph.
        torch = self.torch
        codec, stored = self.nvfp4_weight()
        for activations in ("nvfp4", "bf16"):
            plugin = self.assemble(options={"activations": activations})["projections"]
            m = plugin.load(stored, "nvfp4")
            x = self.x[:1].to(torch.bfloat16).clone()
            eager = plugin.matmul(x, m)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                plugin.matmul(x, m)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = plugin.matmul(x, m)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(bool((out == eager).all()), activations)


class DraftEncodingTests(PluginTestCase):
    """Draft-only re-encoding: torch operations at load that give nvfp4.py's codes and scales exactly (on CUDA, or
    torch's CPU without it), the matrix multiplies as the codec's values, and the facts record each group."""

    def setUp(self):
        self.where = self.device()
        import torch
        from crit_plugins.assemble import load_module
        self.torch = torch
        module = load_module(self.registry()["quantized-projections"])
        self.matrices, self.codec = module.device_matrices, module.nvfp4

    def weight(self):
        """40 x 96 BF16 values: g = 2^10 exactly (amax 2.625), a block of E2M1 ties (6, then every midpoint, both
        signs), two blocks whose scale ties between E4M3 neighbours, a zero block with a negative zero, the rest
        random."""
        torch = self.torch
        unit = 2.0 ** -10
        w = torch.randn(40, 96, generator=torch.Generator().manual_seed(3)) * 0.05
        w[0, :16] = 0
        w[0, 5] = 2.625
        ties = [6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 0.0]
        w[1, :16] = torch.tensor(ties) * unit
        w[2, :16] = torch.tensor([6.375] + [1.0] * 15) * unit    # scale 1.0625: ties 1.0 and 1.125, even code 56
        w[2, 16:32] = torch.tensor([7.125] + [-1.0] * 15) * unit  # scale 1.1875: ties 1.125 and 1.25, even code 58
        w[3, :16] = 0
        w[3, 7] = -0.0
        w = w.to(torch.bfloat16)
        self.assertEqual(float(w[2, 0]), 6.375 * unit)
        return w

    def rows(self, values):
        return [[float(v) for v in row] for row in values.double().cpu().tolist()]

    def test_nvfp4_re_encoding_gives_the_codec_s_codes_and_scales_from_bf16_and_fp8(self):
        torch = self.torch
        w = self.weight()
        scale = (w.float().abs().amax(dim=1) / 448).clamp(min=1e-8)
        fp8 = (w.float() / scale[:, None]).to(torch.float8_e4m3fn)
        for stored, format, values in (({"weight": w}, "bf16", w.double()),
                                       ({"weight": fp8, "scale": scale}, "fp8:channel",
                                        fp8.double() * scale.double()[:, None]),
                                       ({"weight": fp8, "scale": scale.max()}, "fp8:tensor",
                                        fp8.double() * float(scale.max()))):
            found, kind = self.matrices.reencode(stored, format, "nvfp4", self.where)
            self.assertEqual(kind, "nvfp4")
            g = float(found["global_scale"])
            amax = float(values.abs().max())
            self.assertEqual(g, fp32(self.codec.global_scale([amax])), format)
            codes, scales = found["weight"].cpu().tolist(), found["scale"].cpu().tolist()
            for n, row in enumerate(self.rows(values)):
                want_codes, want_scales = self.codec.quantize_row(row, g)
                self.assertEqual(bytes(codes[n]), self.codec.pack(want_codes), (format, n))
                self.assertEqual(bytes(scales[n]), bytes(want_scales), (format, n))
            if format == "bf16":
                self.assertEqual(g, 1024.0)
                self.assertEqual(scales[1][0], 56)                       # 6 * 2^-10 * g / 6 = 1.0 exactly
                self.assertEqual(scales[2][:2], [56, 58])                # the E4M3 ties go to the even code
                self.assertEqual(codes[1][:8], list(self.codec.pack(
                    [7, 0, 2, 2, 4, 4, 6, 6, 0, 10, 10, 12, 12, 14, 14, 0])))  # E2M1 ties to even, both signs
                self.assertEqual(codes[3][:8], [0] * 8)                   # a zero block: no negative-zero codes

    def test_fp8_re_encoding_gives_e4m3_codes_and_a_row_scale(self):
        torch = self.torch
        w = self.weight()
        found, kind = self.matrices.reencode({"weight": w}, "bf16", "fp8", self.where)
        self.assertEqual(kind, "fp8:channel")
        codes, scale = found["weight"].view(torch.uint8).cpu().tolist(), found["scale"].cpu().tolist()
        for n, row in enumerate(self.rows(w)):
            amax = max(abs(v) for v in row)
            self.assertEqual(scale[n], fp32(amax / 448) if amax else 1.0, n)
            self.assertEqual(codes[n], [self.codec.e4m3_encode(v / scale[n]) for v in row], n)

    def test_re_encoding_refuses_what_it_cannot_shrink(self):
        torch = self.torch
        w = self.weight()
        with self.assertRaisesRegex(ValueError, "does not re-encode to fp8"):
            self.matrices.reencode({"weight": w[:, :32].to(torch.uint8), "scale": torch.ones(40, 2)}, "nvfp4", "fp8")
        with self.assertRaisesRegex(ValueError, "re-encodes to nvfp4, fp8, not 'int4'"):
            self.matrices.reencode({"weight": w}, "bf16", "int4")
        with self.assertRaisesRegex(ValueError, "multiple of 16 columns"):
            self.matrices.reencode({"weight": w[:, :40]}, "bf16", "nvfp4")
        bad = w.clone()
        bad[4, 4] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.matrices.reencode({"weight": bad}, "bf16", "nvfp4")

    def test_a_draft_only_matrix_multiplies_as_its_codec_values_and_the_facts_record_the_group(self):
        torch = self.torch
        w = self.weight().to(self.where)
        plugin = self.assemble(options={"activations": "nvfp4"})["projections"]
        m = plugin.load({"weight": w}, "bf16", role="mtp.gate_up", encode="nvfp4")
        plugin.load({"weight": w}, "bf16", role="mtp.gate_up", encode="nvfp4")
        head = plugin.load({"weight": w[:32]}, "bf16", role="draft_head", encode="fp8")   # cuBLASLt: rows % 16
        self.assertEqual((m.format, m.activations), ("nvfp4", "bf16"))   # W4A16: no calibrated activation scale
        x = torch.randn(9, 96, generator=torch.Generator().manual_seed(2)).to(self.where).to(torch.bfloat16)
        exact = x.double() @ self.matrices.dense(m).double().t()
        found = plugin.matmul(x, m).double()
        self.assertLess(float((found - exact).abs().max() / exact.abs().max()), 1e-2)
        for rows in (2, 5, 8):
            batch = plugin.matmul(x[:rows], m)
            for r in range(rows):
                self.assertTrue(torch.equal(batch[r], plugin.matmul(x[r:r + 1].clone(), m)[0]), (rows, r))
        if self.where == "cuda":   # FP8 through torch's scaled matmul (cuBLASLt)
            self.assertEqual(tuple(plugin.matmul(x, head).shape), (9, 32))
        facts = plugin.facts()
        self.assertEqual(facts["draft_encoding"], {
            "mtp.gate_up": {"stored": "bf16", "encoded": "nvfp4", "matrices": 2, "stored_bytes": 2 * 40 * 96 * 2,
                            "encoded_bytes": 2 * (40 * 48 + 40 * 6 + 4)},
            "draft_head": {"stored": "bf16", "encoded": "fp8:channel", "matrices": 1, "stored_bytes": 32 * 96 * 2,
                           "encoded_bytes": 32 * 96 + 32 * 4}})
        self.assertNotIn("prompt_activations", facts)    # the target's prompts are priced by the target's matrices
        with self.assertRaisesRegex(ValueError, "re-encodes bf16 to nvfp4, not bf16 to fp8:channel"):
            plugin.load({"weight": w}, "bf16", role="mtp.gate_up", encode="fp8")
        # The same role's matrices stored in another format are a group of their own.
        fp8 = plugin.load(self.matrices.reencode({"weight": w}, "bf16", "fp8")[0], "fp8:channel", role="mtp.gate_up",
                          encode="nvfp4")
        self.assertEqual(fp8.format, "nvfp4")
        self.assertEqual(plugin.facts()["draft_encoding"]["mtp.gate_up:fp8:channel"]["encoded"], "nvfp4")
        stored = plugin.load({"weight": w}, "bf16", role="mtp.norm", encode="bf16")   # nothing to re-encode
        self.assertEqual(stored.format, "bf16")
        self.assertEqual(plugin.facts()["draft_encoding"]["mtp.norm"],   # draft-only all the same, as stored
                         {"stored": "bf16", "encoded": "bf16", "matrices": 1, "stored_bytes": 40 * 96 * 2,
                          "encoded_bytes": 40 * 96 * 2})


if __name__ == "__main__":
    unittest.main()
