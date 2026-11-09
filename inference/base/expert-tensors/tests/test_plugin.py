from crit_plugins.slots import expert_weights


class ExpertSourceChecks:
    """What every expert_source must do; the test class gives assemble()."""

    def test_it_has_the_expert_source_interface(self):
        self.assertImplements()

    def test_matrices_are_the_ones_added(self):
        source = self.assemble()["expert_source"]
        for layer, expert in ((0, 0), (0, 5), (3, 2)):
            source.add(layer, expert, *expert_weights(layer, expert))
        for layer, expert in ((3, 2), (0, 0), (0, 5), (0, 0)):
            gate_up, down = source.matrices(layer, expert)
            self.assertEqual((gate_up, down), expert_weights(layer, expert))

    def test_a_bad_shape_an_expert_added_twice_and_a_missing_one_are_refused(self):
        source = self.assemble()["expert_source"]
        gate_up, down = expert_weights(0, 0)
        with self.assertRaisesRegex(ValueError, "gate_up"):
            source.add(0, 0, gate_up[:-1], down)
        with self.assertRaisesRegex(ValueError, "gate_up"):
            source.add(0, 0, gate_up, [row[:-1] for row in down])
        source.add(0, 0, gate_up, down)
        with self.assertRaisesRegex(ValueError, "already"):
            source.add(0, 0, gate_up, down)
        with self.assertRaises(KeyError):
            source.matrices(0, 1)

from crit_plugins.testing import PluginTestCase


class ExpertTensorsTests(ExpertSourceChecks, PluginTestCase):
    pass
