import math

from crit_plugins.slots import CASES, expert_weights
from crit_plugins.testing import PluginTestCase

USE = {"expert_source": "expert-tensors"}


def swiglu(gate_up, down, x):
    inter = len(gate_up) // 2
    act = []
    for i in range(inter):
        g = sum(a * b for a, b in zip(gate_up[i], x))
        u = sum(a * b for a, b in zip(gate_up[inter + i], x))
        act.append(g / (1 + math.exp(-g)) * u)
    return [sum(a * b for a, b in zip(row, act)) for row in down]


class ReferenceExpertsTests(PluginTestCase):
    def test_it_has_the_experts_interface(self):
        self.assertImplements(self.assemble(USE))

    def test_it_is_the_weighted_sum_of_swiglu_experts(self):
        parts = self.assemble(USE)
        for e in range(3):
            parts["expert_source"].add(2, e, *expert_weights(2, e))
        x = [[0.3, -0.2, 0.5, 0.1, -0.4, 0.7], [1.0, 0.0, -1.0, 0.5, 0.5, -0.5]]
        out = parts["experts"].run(2, x, [[0, 2], [1, -1]], [[0.75, 0.25], [1.0, 0.6]])
        first = [0.75 * a + 0.25 * b for a, b in zip(swiglu(*expert_weights(2, 0), x[0]),
                                                      swiglu(*expert_weights(2, 2), x[0]))]
        self.assertClose(out[0], first, atol=1e-12)
        self.assertClose(out[1], swiglu(*expert_weights(2, 1), x[1]), atol=1e-12)

    def test_a_token_without_routes_is_zero_and_shapes_are_checked(self):
        experts = self.assemble(USE)["experts"]
        self.assertEqual(experts.run(0, [[1.0, 2.0]], [[-1]], [[1.0]]), [[0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "same number"):
            experts.run(0, [[1.0], [2.0]], [[-1], [-1, -1]], [[1.0], [1.0, 1.0]])

    def test_the_cases_run(self):
        out = CASES["experts"](self.assemble(USE))
        self.assertEqual([len(out["layer0"]), len(out["layer1"]), len(out["again"])], [5, 3, 3])
        self.assertEqual(out["again"], out["layer0"][2:])
