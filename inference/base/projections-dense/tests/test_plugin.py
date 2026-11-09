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
            found = p.matmul(x[:rows], m, torch.float32)
            self.assertEqual(found.dtype, torch.float32)
            self.assertClose(found.tolist(), (x[:rows].float() @ m.float().t()).tolist(), atol=1e-5, rtol=1e-5)

    def test_it_multiplies(self):
        p = self.assemble()["projections"]
        w = p.prepare([[1, 2], [3, 4], [0, -1]])
        self.assertEqual(p.project(w, [[1.0, 1.0], [2.0, 0.5]]), [[3.0, 7.0, -1.0], [3.0, 8.0, -0.5]])
        self.assertEqual(p.dense(w), [[1.0, 2.0], [3.0, 4.0], [0.0, -1.0]])
