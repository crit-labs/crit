"""grouped-experts' host form is the FP64 routed SwiGLU block its reference computes, and the plugin has the experts
slot's device form, its bundles and its route rows."""
from crit_plugins.slots import CASES, expert_weights
from crit_plugins.testing import PluginTestCase

USE = {"expert_source": "expert-tensors"}


class GroupedExpertsTests(PluginTestCase):
    def test_it_has_the_experts_interface_and_its_extensions(self):
        self.assertImplements(self.assemble(USE))

    def test_its_host_form_is_its_reference_on_the_cases(self):
        self.assertMatchesReference(USE, atol=1e-12, rtol=0.0)

    def test_a_token_without_routes_is_zero(self):
        experts = self.assemble(USE)["experts"]
        self.assertEqual(experts.run(0, [[1.0, 2.0]], [[-1]], [[1.0]]), [[0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "same number"):
            experts.run(0, [[1.0], [2.0]], [[-1], [-1, -1]], [[1.0], [1.0, 1.0]])

    def test_the_cases_run(self):
        parts = self.assemble(USE)
        out = CASES["experts"](parts)
        self.assertEqual(out["again"], out["layer0"][2:])
        self.assertEqual(len(expert_weights(0, 0)[0]), 8)
