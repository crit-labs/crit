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

    def test_prompt_auto_follows_the_level_the_engine_asks_of_its_prompt_rows(self):
        self.addCleanup(kernels.use_exactness, None)
        for asked, prompt in ((None, "dequantize"), ("repeatable", "dequantize"), ("row-exact", "exact"),
                              ("exact", "exact"), ("declared", "exact")):
            kernels.use_exactness(asked and {"decode": "row-exact", "prompt": asked})
            plugin = self.assemble()["projections"]
            self.assertEqual(plugin.prompt, prompt, asked)
            self.assertEqual(plugin.exactness(), {})   # no matrix loaded: it states no level yet
        kernels.use_exactness("exact")   # an option set to one path is that path, whatever is asked
        self.assertEqual(self.assemble(options={"prompt": "dequantize"})["projections"].prompt, "dequantize")

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
            y = plugin.matmul(x, matrix, prompt=False)
            # Each row's bits its own, alone and among others.
            self.assertTrue(torch.equal(y[:1], plugin.matmul(x[:1], matrix, prompt=False)), name)
            values = torch.from_numpy(stored_values(raw, type_id, k)).to("cuda", torch.bfloat16)
            if plugin.kernel is not None:
                self.assertTrue(torch.equal(y, plugin.kernel.gemv(x, matrix.kernel)), name)
                self.assertTrue(torch.equal(plugin.kernel.dequantize(matrix.kernel), values), name)
            # Prompt rows by dequantize: the stored values in BF16 through cuBLAS.
            p = self.activations(40, k)
            want = torch.mm(p, values.t(), out_dtype=torch.float32).to(torch.bfloat16)
            self.assertTrue(torch.equal(plugin.matmul(p, matrix, prompt=True), want), name)
            # cuBLAS's bits follow its row count: a 3-row prompt call against cuBLAS on the same 3 rows.
            want3 = torch.mm(p[:3], values.t(), out_dtype=torch.float32).to(torch.bfloat16)
            self.assertTrue(torch.equal(plugin.matmul(p[:3], matrix, prompt=True), want3), name)
        self.assertEqual(set(plugin.facts()["gguf_formats"]), set(TYPES))
        # What its rows keep: the kernel's declared order for decode rows, cuBLAS's row count for dequantized prompts.
        self.assertEqual(plugin.exactness(), {"decode": "declared" if plugin.kernel is not None else "row-exact",
                                              "prompt": "repeatable"})
        # The shapes crit kernels sweep --matrices tunes: every matrix as loaded.
        self.assertEqual(plugin.facts()["matrices"], [{"N": 48, "K": 512, "format": name} for name in sorted(TYPES)])

    def test_exact_prompt_rows_have_the_decode_rows_bits(self):
        torch = self.torch
        plugin = self.plugin(prompt="exact")
        if plugin.kernel is None:
            self.skipTest("gguf-rows is not built here")
        x = self.activations(20, 1024)
        for type_id, name in ((12, "Q4_K"), (35, "TQ2_0")):
            raw = blocks(type_id, 32, 1024, seed=3)
            matrix = plugin.load({"weight": raw}, f"gguf:{name}")
            prompt = plugin.matmul(x, matrix, prompt=True)
            self.assertTrue(torch.equal(prompt[:8], plugin.matmul(x[:8], matrix, prompt=False)), name)
        self.assertEqual(plugin.facts()["row_invariance"], {"rows": 8, "prompt_rows": True, "prompt_exact": True})
        self.assertEqual(plugin.exactness(), {"decode": "declared", "prompt": "declared"})

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
            self.assertTrue(torch.equal(y[t:t + 1], plugin.matmul(x[t:t + 1], one, prompt=False)), t)

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
        y = plugin.matmul(x, matrix, prompt=False)
        self.assertTrue(torch.equal(y[1:2], plugin.matmul(x[1:2], matrix, prompt=False)))
        self.assertEqual(plugin.facts()["prompt"], "fallback")
        self.assertEqual(plugin.exactness(), {"decode": "row-exact", "prompt": "repeatable"})

    def test_dense_tensors_load_as_bf16_and_other_formats_are_refused(self):
        torch = self.torch
        plugin = self.plugin()
        w = torch.randn(8, 64)
        matrix = plugin.load({"weight": w}, "bf16")
        x = self.activations(2, 64)
        self.assertTrue(torch.equal(plugin.matmul(x[:1], matrix, prompt=False),
                                    torch.mm(x[:1], w.to("cuda", torch.bfloat16).t(), out_dtype=torch.float32)
                                    .to(torch.bfloat16)))
        with self.assertRaisesRegex(ValueError, "gguf:<TYPE>"):
            plugin.load({"weight": blocks(8, 4, 32, seed=1)}, "gguf:Q8_1")
        with self.assertRaisesRegex(ValueError, "144-byte blocks"):
            plugin.load({"weight": blocks(12, 4, 256, seed=1)[:, :100]}, "gguf:Q4_K")
        with self.assertRaisesRegex(ValueError, "keeps matrices as stored"):
            plugin.load({"weight": w}, "bf16", encode="nvfp4")


def gguf_file(path, tensors):
    """A GGUF file at `path` of [(name, dtype, shape, data bytes)], written by crit's fixture writer
    (tests/gguf_writer.py in the crit checkout); None where that writer is not beside this plugin."""
    import sys
    from pathlib import Path
    writer = Path(__file__).resolve().parents[5] / "tests"
    if not (writer / "gguf_writer.py").is_file():
        return None
    sys.path.insert(0, str(writer))
    try:
        from gguf_writer import gguf_bytes
    finally:
        sys.path.remove(str(writer))
    path.write_bytes(gguf_bytes([("general.architecture", "llama")], tensors))
    return path


class DenseRowTests(PluginTestCase):
    """A GGUF file's dense matrices (F32, F16, BF16) in an engine that asks exact rows, as preserve mode does: on the
    device the tests run on (the CPU here without CUDA: the one-row-a-call path; on CUDA mma-rows' tiles where it
    builds), every row's bits are the same in any composition of decode calls and prompt chunks, and the plugin keeps
    the level the configuration asks, so Core's loader takes the engine."""

    def setUp(self):
        import torch
        self.torch = torch
        self.at = self.device()
        self.addCleanup(kernels.use_exactness, None)

    def ask(self, mode):
        from crit_plugins.exactness import for_quality_mode
        kernels.use_exactness(for_quality_mode(mode))
        return for_quality_mode(mode)

    def file_matrices(self, plugin):
        """The dense matrices of a seeded GGUF file (F32 [48][2560], F16 [16][512], BF16 [64][256]) and one Q8_0
        matrix, loaded as a model plugin loads them (model-template's GgufMatrices: dense values as bf16, stored rows
        as gguf:<TYPE>)."""
        import tempfile
        from pathlib import Path
        from crit_core.model.gguf import inspect_gguf_model
        from crit_core.model.gguf_blocks import GgufTensors
        torch = self.torch
        g = np.random.default_rng(11)
        f32 = g.standard_normal((48, 2560)).astype(np.float32)
        f16 = g.standard_normal((16, 512)).astype(np.float16)
        bf16 = (g.standard_normal((64, 256)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        q8 = blocks(8, 32, 512, seed=12)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        path = gguf_file(folder / "dense.gguf", [("router.weight", "F32", f32.shape, f32.tobytes()),
                                                 ("alpha.weight", "F16", f16.shape, f16.tobytes()),
                                                 ("indexer.weight", "BF16", bf16.shape, bf16.tobytes()),
                                                 ("up.weight", "Q8_0", (32, 512), q8.tobytes())])
        if path is None:
            self.skipTest("crit's GGUF fixture writer (tests/gguf_writer.py) is not beside this plugin")
        reader = GgufTensors(inspect_gguf_model(path))
        self.addCleanup(reader.close)
        decoded = {"F32": lambda raw: torch.from_numpy(raw.view(np.float32).copy()),
                   "F16": lambda raw: torch.from_numpy(raw.view(np.float16).astype(np.float32)),
                   # the shifted words viewed as float32 in numpy: on Windows the shift's dtype is numpy.uintc,
                   # which torch.from_numpy refuses
                   "BF16": lambda raw: torch.from_numpy((raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32))}
        found = {}
        for name in ("router.weight", "alpha.weight", "indexer.weight", "up.weight"):
            info, raw = reader.info(name), np.ascontiguousarray(reader.rows(name))
            if info.dtype in decoded:
                weight = decoded[info.dtype](raw).to(torch.bfloat16)
                found[name] = plugin.load({"weight": weight}, "bf16")
                self.assertTrue(torch.equal(found[name].dense.cpu(), weight), name)   # the stored values, as BF16
            else:
                found[name] = plugin.load({"weight": torch.from_numpy(raw)}, "gguf:" + info.dtype)
        return found

    def test_a_dense_rows_kernel_follows_its_phase(self):
        # Routing conformance: decode tokens one row a call (or mma-rows' decode tile) at every step of the grid,
        # verify windows past the decode rows too; a prompt chunk's kernel the prompt rows' however short.
        import sys
        from contextlib import contextmanager
        from unittest import mock
        torch = self.torch
        plugin = self.assemble()["projections"]
        matrix = plugin.load({"weight": torch.zeros(16, 256, dtype=torch.bfloat16)}, "bf16")
        module = sys.modules[type(plugin).__module__]
        mm = module._mm

        @contextmanager
        def record():
            calls = []

            def recorded(x16, w):
                calls.append(("one row" if x16.shape[0] == 1 else "a GEMM of rows", x16.shape[0]))
                return mm(x16, w)
            patches = [mock.patch.object(module, "_mm", recorded)]
            tile = getattr(plugin, "tile", None)
            for name in ("mm_rows", "mm_prompt"):
                if tile is not None and callable(getattr(tile, name, None)):
                    real = getattr(tile, name)
                    patches.append(mock.patch.object(tile, name, lambda x, *a, _n=name, _r=real, **k: (
                        calls.append((_n, x.shape[0])), _r(x, *a, **k))[1]))
            for patch in patches:
                patch.start()
            try:
                yield calls
            finally:
                for patch in patches:
                    patch.stop()
        found = self.assertProjectionsRouteByPhase(plugin, matrix, 256, record, device=self.at)
        self.assertEqual(len(found["decode"]), 1)

    def test_dense_rows_keep_their_bits_in_every_composition_where_exact_rows_are_asked(self):
        torch = self.torch
        self.ask("preserve")
        plugin = self.assemble()["projections"]
        self.assertEqual(plugin.prompt, "exact")
        for name, matrix in self.file_matrices(plugin).items():
            if matrix.dense is None:
                continue   # a quantized matrix on gguf-rows: its own tests hold its bits
            k = matrix.columns
            x = torch.randn(40, k, generator=torch.Generator().manual_seed(k)).to(self.at, torch.bfloat16)
            prompt = plugin.matmul(x, matrix, prompt=True)     # one prompt chunk of 40 rows
            alone = torch.cat([plugin.matmul(x[r:r + 1], matrix, prompt=False) for r in range(40)])   # each row decoded alone
            self.assertTrue(torch.equal(prompt, alone), name)
            batched = torch.cat([plugin.matmul(x[r:r + 3], matrix, prompt=False) for r in range(0, 40, 3)])   # decode calls of 3
            self.assertTrue(torch.equal(prompt, batched), name)
            chunks = torch.cat([plugin.matmul(x[:17], matrix, prompt=True),
                                plugin.matmul(x[17:], matrix, prompt=True)])   # other chunks
            self.assertTrue(torch.equal(prompt, chunks), name)
            self.assertTrue(torch.equal(plugin.matmul(x[5:10], matrix, prompt=True), prompt[5:10]), name)
            fp32 = plugin.matmul(x, matrix, out_dtype=torch.float32, prompt=True)
            self.assertTrue(torch.equal(fp32.to(torch.bfloat16), prompt), name)
            # and they are the stored values' products, FP32 sums
            want = torch.mm(x.float(), matrix.dense.float().t())
            self.assertLess(float((fp32 - want).abs().max()), 1e-3 * float(want.abs().max()) + 1e-4, name)

    def test_an_engine_asking_preserve_modes_exactness_takes_a_gguf_with_dense_matrices(self):
        from crit_plugins.assemble import exactness_shortfalls
        asked = self.ask("preserve")
        plugin = self.assemble()["projections"]
        self.file_matrices(plugin)
        kept = plugin.exactness()
        self.assertEqual(kept, {"decode": "row-exact", "prompt": "exact"})
        self.assertEqual(exactness_shortfalls({"asked": asked, "plugins": {"gguf-projections": kept}}), [])
        facts = plugin.facts()
        self.assertEqual(facts["row_invariance"], {"rows": 8, "prompt_rows": True, "prompt_exact": True})
        self.assertTrue(set(facts["dense_paths"]) <= {"tile", "rows"}, facts)
        # dequantize keeps the dense prompt rows cuBLAS's (repeatable): what preserve mode refused
        dequantized = self.assemble(options={"prompt": "dequantize"})["projections"]
        self.file_matrices(dequantized)
        self.assertEqual(dequantized.exactness()["prompt"], "repeatable")
        short = exactness_shortfalls({"asked": asked, "plugins": {"gguf-projections": dequantized.exactness()}})
        self.assertEqual(short, ["gguf-projections keeps its prompt rows repeatable where the configuration's "
                                 "exactness asks exact"])
        self.assertEqual(dequantized.facts()["dense_paths"], ["torch"])

    def test_on_mma_rows_decode_rows_take_its_decode_tile_and_prompt_rows_its_prompt_tile(self):
        # The wiring of the tile path, on a stand-in for mma-rows that computes a row alone in FP32 (the kernel's
        # own bits are mma-rows' tests, on CUDA): decode calls go to mm_rows with the matrix's plan (its K splits, with
        # the plugin's partial sums and counters), prompt calls to mm_prompt at the same splits, a split count whose
        # partial sums pass PARTIAL_BYTES is cut, and a matrix whose K the kernel does not take stays one row a call.
        torch = self.torch
        calls = []

        class Tile:
            @staticmethod
            def eligible(rows, columns):
                return columns % 32 == 0

            @staticmethod
            def plan(rows, columns, weights, device=None, acts="bf16"):
                return 2, 4, 7

            @staticmethod
            def partial_elements(tokens, rows, splits):
                return splits * tokens * rows if splits > 1 else 0

            @staticmethod
            def row_tiles(rows, groups, warps):
                return -(-rows // (16 * groups * warps))

            @staticmethod
            def _rows(x, w, out):
                for r in range(x.shape[0]):
                    out[r] = torch.mv(w.float(), x[r].float())
                return out

            @classmethod
            def mm_rows(cls, x, w, out, weights, groups=1, warps=4, splits=1, partial=None, counters=None):
                assert splits == 1 or partial.numel() >= cls.partial_elements(x.shape[0], w.shape[0], splits)
                assert counters is not None and counters.numel() >= cls.row_tiles(w.shape[0], 1, warps)
                calls.append(("mm_rows", weights, groups, warps, splits, x.shape[0]))
                return cls._rows(x, w, out)

            @classmethod
            def mm_prompt(cls, x, w, out, weights, splits=1):
                calls.append(("mm_prompt", weights, splits, x.shape[0]))
                return cls._rows(x, w, out)

        self.ask("preserve")
        plugin = self.assemble()["projections"]
        plugin.tile = Tile
        tiled = plugin.load({"weight": torch.randn(24, 64)}, "bf16")
        odd = plugin.load({"weight": torch.randn(8, 48)}, "bf16")
        self.assertEqual((tiled.path, tiled.tile, odd.path), ("tile", (2, 4, 7), "rows"))
        # 80,000 rows in 7 splits would take 17.9 MB of partial sums a decode call: 6 splits keep within 16 MiB
        wide = plugin.load({"weight": torch.zeros(80000, 32)}, "bf16")
        self.assertEqual(wide.tile, (2, 4, 6))
        x = torch.randn(20, 64).to(self.at, torch.bfloat16)
        self.assertTrue(torch.equal(plugin.matmul(x, tiled, prompt=True),
                                    torch.cat([plugin.matmul(x[r:r + 1], tiled, prompt=False)
                                               for r in range(20)])))
        self.assertEqual(calls[0], ("mm_prompt", "bf16", 7, 20))
        self.assertEqual(calls[1], ("mm_rows", "bf16", 2, 4, 7, 1))
        plugin.matmul(x[:4], tiled, prompt=True)
        self.assertEqual(calls[-1], ("mm_prompt", "bf16", 7, 4))
        plugin.matmul(torch.randn(3, 48).to(self.at, torch.bfloat16), odd, prompt=False)
        self.assertEqual(len(calls), 22)   # the 48-column matrix took no tile call
        self.assertEqual(plugin.facts()["dense_paths"], ["rows", "tile"])


if __name__ == "__main__":
    unittest.main()
