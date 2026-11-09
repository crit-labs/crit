"""fused-norms' host form is the norms slot's FP64 rows: it equals torch-norms on crit's cases, and it names the
registry kernel it loads."""
from crit_plugins.testing import PluginTestCase


class FusedNormsHostTests(PluginTestCase):
    def test_it_has_the_norms_interface_and_its_device_form(self):
        self.assertImplements()

    def test_it_matches_torch_norms_on_the_norms_cases(self):
        self.assertMatchesReference(atol=1e-12, rtol=1e-12)

    def test_it_declares_the_registry_kernel_it_loads_and_its_fallback(self):
        (kernel,) = self.registry()["fused-norms"]["kernels"]
        self.assertEqual((kernel["name"], kernel["version"], kernel["revision"]), ("crit-labs/fused-norms", 1, None))
        self.assertIn("torch-norms", kernel["fallback"])
