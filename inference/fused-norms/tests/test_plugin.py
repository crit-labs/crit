"""fused-norms' host form is the norms slot's FP64 rows: it equals torch-norms on crit's cases, and it names crit's
kernel it loads."""
from crit_plugins.testing import PluginTestCase


class FusedNormsHostTests(PluginTestCase):
    def test_it_has_the_norms_interface_and_its_device_form(self):
        self.assertImplements()

    def test_it_matches_torch_norms_on_the_norms_cases(self):
        self.assertMatchesReference(atol=1e-12, rtol=1e-12)

    def test_it_declares_the_load_kernel_it_loads_and_its_fallback(self):
        (kernel,) = self.registry()["fused-norms"]["kernels"]
        self.assertEqual(kernel["name"], "fused-norms")
        self.assertIn("torch-norms", kernel["fallback"])

    def test_an_engine_that_asks_exact_prompt_rows_loads_with_it(self):
        # On the kernel a prompt row has the bits of the same row decoded (exact); the torch fallback states nothing.
        # Either way Core finds no shortfall against exact prompt rows.
        from crit_plugins.assemble import exactness_shortfalls
        norms = self.assemble()["norms"]
        stated = norms.exactness()
        self.assertEqual(stated, {"decode": "row-exact", "prompt": "exact"} if norms.kernel is not None else {})
        report = {"asked": {"decode": "row-exact", "prompt": "exact"}, "plugins": {"fused-norms": stated}}
        self.assertEqual(exactness_shortfalls(report), [])
