"""gguf-projections: the host form (Q8_0 blocks of the slot's dense cases, exact sums over the stored values) and, on
CUDA, the device form through crit's kernel gguf-rows where it builds here and through the host-decoded BF16 fallback:
every format's matrix loaded from its stored rows, decode rows with each row's bits its own (the kernel's gemv bits),
prompt rows by dequantize (cuBLAS on the stored values in BF16) and by exact (the decode rows' bits), expert stacks,
and the formats it refuses."""
import unittest
from unittest import mock

import numpy as np

from crit_core.model.gguf_blocks import BLOCKS, SCALE_BYTES, TYPES, stored_values
from crit_plugins import kernels
from crit_plugins.testing import PluginTestCase


def blocks(type_id, n, k, seed, experts=None):
    """Seeded stored rows uint8 [N][row bytes] (or [experts][N][row bytes]) with finite scales."""
    values, size = BLOCKS[type_id]
    g = np.random.default_rng(seed)
    shape = (experts or 1, n, k // values, size)
    raw = g.integers(0, 256, size=shape, dtype=np.uint8)
    for at in SCALE_BYTES[type_id]:
        scale = (g.uniform(1, 2, size=shape[:-1]) * 2.0 ** g.integers(-12, -4, size=shape[:-1])).astype(np.float16)
        raw[..., at:at + 2] = scale.view(np.uint8).reshape(*shape[:-1], 2)
    if type_id == 29:   # IQ1_M's FP16 d: the top nibbles of its four scale words
        bits = (g.uniform(1, 2, size=shape[:-1]) * 2.0 ** g.integers(-12, -4, size=shape[:-1])).astype(
            np.float16).view(np.uint16)
        for u in range(4):
            raw[..., 49 + 2 * u] = (raw[..., 49 + 2 * u] & 15) | (((bits >> (4 * u)) & 15).astype(np.uint8) << 4)
    if type_id == 39:   # MXFP4's E8M0 byte: modest powers of two
        raw[..., 0] = g.integers(115, 126, size=shape[:-1], dtype=np.uint8)
    raw = raw.reshape(experts or 1, n, -1)
    return raw if experts else raw[0]


class HostFormTests(PluginTestCase):
    def test_it_has_the_projections_interface(self):
        for options in ({}, {"prompt": "exact"}):
            self.assertImplements(self.assemble(options=options))

    def test_it_projects_its_stored_values_as_the_dense_projection_does(self):
        dense = self.reference()["projections"]
        plugin = self.assemble()["projections"]
        rows = [[((i * 13 + j * 5) % 17 - 8) / 8 for j in range(64)] for i in range(5)]
        x = [[((t * 3 + j) % 9 - 4) / 4 for j in range(64)] for t in range(3)]
        w = plugin.prepare(rows)
        self.assertClose(plugin.project(w, x), dense.project(dense.prepare(plugin.dense(w)), x), atol=1e-12, rtol=1e-12)

    def test_q8_0_blocks_of_a_dense_matrix_and_exact_sums_over_their_values(self):
        plugin = self.assemble()["projections"]
        rows = [[(i * 7 + j * 3) % 11 - 5.5 for j in range(64)] for i in range(3)]
        weight = plugin.prepare(rows)
        values = plugin.dense(weight)
        for row, got in zip(rows, values):
            for block in range(2):
                d = np.float32(np.float16(max(abs(v) for v in row[32 * block:32 * block + 32]) / 127))
                for a, b in zip(row[32 * block:32 * block + 32], got[32 * block:32 * block + 32]):
                    self.assertLessEqual(abs(a - b), float(d) / 2 + 1e-6)
        x = [[0.5] * 64, [(-1) ** j for j in range(64)]]
        want = [[sum(a * b for a, b in zip(r, xr)) for r in values] for xr in x]
        for got_row, want_row in zip(plugin.project(weight, x), want):
            self.assertClose(got_row, want_row, atol=1e-9, rtol=1e-12)


class DeviceTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("gguf-projections runs on CUDA")
        import torch
        self.torch = torch

    def plugin(self, **options):
        return self.assemble(options=options)["projections"]

    def activations(self, rows, k):
        torch = self.torch
        return torch.randn(rows, k, generator=torch.Generator().manual_seed(rows + k)).to("cuda", torch.bfloat16)

    def test_every_format_loads_from_its_stored_rows_and_decode_rows_are_the_kernels_bits(self):
        torch = self.torch
        plugin = self.plugin()
        record = kernels.loaded()["gguf-rows"]
        self.assertIn(record["source"], kernels.SOURCES)
        for name, type_id in TYPES.items():
            n, k = 48, 512
            raw = blocks(type_id, n, k, seed=type_id)
            matrix = plugin.load({"weight": raw}, f"gguf:{name}")
            self.assertEqual((matrix.rows, matrix.columns), (n, k))
            x = self.activations(5, k)
            y = plugin.matmul(x, matrix)
            # Each row's bits its own, alone and among others.
            self.assertTrue(torch.equal(y[:1], plugin.matmul(x[:1], matrix)), name)
            values = torch.from_numpy(stored_values(raw, type_id, k)).to("cuda", torch.bfloat16)
            if plugin.kernel is not None:
                self.assertTrue(torch.equal(y, plugin.kernel.gemv(x, matrix.kernel)), name)
                self.assertTrue(torch.equal(plugin.kernel.dequantize(matrix.kernel), values), name)
            # Prompt rows by dequantize: the stored values in BF16 through cuBLAS.
            p = self.activations(40, k)
            want = torch.mm(p, values.t(), out_dtype=torch.float32).to(torch.bfloat16)
            self.assertTrue(torch.equal(plugin.matmul(p, matrix), want), name)
            # cuBLAS's bits follow its row count: a 3-row prompt call against cuBLAS on the same 3 rows.
            want3 = torch.mm(p[:3], values.t(), out_dtype=torch.float32).to(torch.bfloat16)
            self.assertTrue(torch.equal(plugin.matmul(p[:3], matrix, prompt=True), want3), name)
        self.assertEqual(set(plugin.facts()["gguf_formats"]), set(TYPES))
        # The shapes crit kernels sweep --matrices tunes: every matrix as loaded.
        self.assertEqual(plugin.facts()["matrices"], [{"N": 48, "K": 512, "format": name} for name in sorted(TYPES)])

    def test_exact_prompt_rows_have_the_decode_rows_bits(self):
        torch = self.torch
        plugin = self.plugin(prompt="exact")
        if plugin.kernel is None:
            self.skipTest("gguf-rows is not built here")
        raw = blocks(12, 32, 1024, seed=3)
        matrix = plugin.load({"weight": raw}, "gguf:Q4_K")
        x = self.activations(20, 1024)
        prompt = plugin.matmul(x, matrix)
        self.assertTrue(torch.equal(prompt[:8], plugin.matmul(x[:8], matrix)))
        self.assertEqual(plugin.facts()["row_invariance"], {"rows": 8, "prompt_rows": True, "prompt_exact": True})

    def test_an_expert_stacks_rows_each_have_the_plain_calls_bits(self):
        torch = self.torch
        plugin = self.plugin()
        raw = blocks(23, 16, 512, seed=4, experts=3)
        stack = plugin.load({"weight": raw}, "gguf:IQ4_XS")
        x = self.activations(4, 512)
        ids = torch.tensor([2, 0, 2, 1], device="cuda")
        y = plugin.matmul_experts(x, stack, ids)
        for t, e in enumerate(ids.tolist()):
            one = plugin.load({"weight": raw[e]}, "gguf:IQ4_XS")
            self.assertTrue(torch.equal(y[t:t + 1], plugin.matmul(x[t:t + 1], one)), t)

    def test_without_the_kernel_the_host_decoded_values_run_through_torch(self):
        torch = self.torch
        with mock.patch("crit_plugins.kernels.load_kernel", return_value=None):
            plugin = self.plugin()
        self.assertIsNone(plugin.kernel)
        raw = blocks(21, 24, 256, seed=5)
        matrix = plugin.load({"weight": raw}, "gguf:IQ3_S")
        values = torch.from_numpy(stored_values(raw, 21, 256)).to("cuda", torch.bfloat16)
        self.assertTrue(torch.equal(matrix.dense, values))
        x = self.activations(3, 256)
        y = plugin.matmul(x, matrix)
        self.assertTrue(torch.equal(y[1:2], plugin.matmul(x[1:2], matrix)))
        self.assertEqual(plugin.facts()["prompt"], "fallback")

    def test_dense_tensors_load_as_bf16_and_other_formats_are_refused(self):
        torch = self.torch
        plugin = self.plugin()
        w = torch.randn(8, 64)
        matrix = plugin.load({"weight": w}, "bf16")
        x = self.activations(2, 64)
        self.assertTrue(torch.equal(plugin.matmul(x[:1], matrix),
                                    torch.mm(x[:1], w.to("cuda", torch.bfloat16).t(), out_dtype=torch.float32)
                                    .to(torch.bfloat16)))
        with self.assertRaisesRegex(ValueError, "gguf:<TYPE>"):
            plugin.load({"weight": blocks(8, 4, 32, seed=1)}, "gguf:TQ1_0")
        with self.assertRaisesRegex(ValueError, "144-byte blocks"):
            plugin.load({"weight": blocks(12, 4, 256, seed=1)[:, :100]}, "gguf:Q4_K")
        with self.assertRaisesRegex(ValueError, "keeps matrices as stored"):
            plugin.load({"weight": w}, "bf16", encode="nvfp4")


if __name__ == "__main__":
    unittest.main()
