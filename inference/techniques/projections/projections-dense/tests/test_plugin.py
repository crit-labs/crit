from crit_plugins.testing import PluginTestCase


class DenseProjectionsTests(PluginTestCase):
    def test_it_has_the_projections_interface(self):
        self.assertImplements()

    def test_its_device_form_gives_fp32_logits_on_the_cpu_too(self):
        # torch has no CPU kernel for a BF16 GEMM into FP32 (aten::mm.dtype): the CPU path multiplies the BF16 values
        # in FP32, the same products and FP32 sums.
        self.device()
        import torch
        p = self.assemble()["projections"]
        g = torch.Generator().manual_seed(3)
        m = p.load({"weight": torch.randn(24, 64, generator=g)}, "bf16")
        x = torch.randn(11, 64, generator=g).to(torch.bfloat16)
        for rows in (1, 5, 11):
            found = p.matmul(x[:rows], m, torch.float32, prompt=rows > 8)   # 11 rows: a prompt chunk
            self.assertEqual(found.dtype, torch.float32)
            self.assertClose(found.tolist(), (x[:rows].float() @ m.float().t()).tolist(), atol=1e-5, rtol=1e-5)

    def test_it_multiplies(self):
        p = self.assemble()["projections"]
        w = p.prepare([[1, 2], [3, 4], [0, -1]])
        self.assertEqual(p.project(w, [[1.0, 1.0], [2.0, 0.5]]), [[3.0, 7.0, -1.0], [3.0, 8.0, -0.5]])
        self.assertEqual(p.dense(w), [[1.0, 2.0], [3.0, 4.0], [0.0, -1.0]])

    def test_a_decode_rows_call_follows_its_phase(self):
        # Routing conformance: decode tokens one row a call (each its own bits) at every step of the grid, verify
        # windows past the decode rows too; a prompt chunk one GEMM of its rows (row_invariance prompt_rows false:
        # its kernel may follow the chunk's length).
        from contextlib import contextmanager
        from unittest import mock
        import torch
        p = self.assemble()["projections"]
        m = p.load({"weight": torch.zeros(8, 32)}, "bf16")
        linear = torch.nn.functional.linear

        @contextmanager
        def record():
            calls = []

            def recorded(x, w, *args, **kwargs):
                calls.append(("one row" if x.shape[0] == 1 else "a GEMM of rows", x.shape[0]))
                return linear(x, w, *args, **kwargs)
            with mock.patch("torch.nn.functional.linear", recorded):
                yield calls
        found = self.assertProjectionsRouteByPhase(p, m, 32, record)
        self.assertEqual(list(found["decode"]), [("one row",)])
