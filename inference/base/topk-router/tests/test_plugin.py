import math

from crit_plugins.slots import CASES
from crit_plugins.testing import PluginTestCase


class TopKRouterTests(PluginTestCase):
    def test_it_has_the_router_interface(self):
        self.assertImplements()

    def test_softmax_over_the_selected_logits(self):
        router = self.assemble()["router"]
        (ids, weights), = router.route([[1.0, 3.0, 2.0, 0.0]], 2)
        self.assertEqual(ids, [1, 2])
        self.assertAlmostEqual(weights[0], 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_a_correction_bias_changes_the_selection_not_the_weights(self):
        router = self.assemble(options={"scoring": "sigmoid", "score_before_selection": True})["router"]
        (ids, weights), = router.route([[1.0, 3.0, 2.0, 0.0]], 1, bias=[0.0, 0.0, 0.0, 5.0])
        self.assertEqual(ids, [3])
        self.assertAlmostEqual(weights[0], 1 / (1 + math.exp(-0.0)))

    def test_the_router_cases_run(self):
        out = CASES["router"](self.assemble())
        self.assertEqual([len(ids) for ids, _ in out["top3"]], [3] * 5)
