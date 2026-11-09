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

import os

from crit_plugins.testing import PluginTestCase


class SSDExpertMaterializationTests(ExpertSourceChecks, PluginTestCase):
    def test_the_experts_cases_equal_the_in_memory_store_exactly(self):
        for host_bytes in (0, 200, 1 << 20):
            with self.subTest(host_bytes=host_bytes):
                self.assertMatchesReference({"experts": "experts-reference"}, {"host_bytes": host_bytes},
                                            cases=["experts"], atol=0.0, rtol=0.0)

    def test_the_host_tier_serves_a_second_read_and_a_read_freezes_the_file(self):
        source = self.assemble()["expert_source"]
        source.add(0, 0, *expert_weights(0, 0))
        source.add(0, 1, *expert_weights(0, 1))
        source.matrices(0, 0)
        source.matrices(0, 0)
        self.assertEqual(source.tier.stats["hits"], 1)
        with self.assertRaisesRegex(ValueError, "immutable"):
            source.add(0, 2, *expert_weights(0, 2))

    def test_a_changed_file_is_refused_and_close_removes_it(self):
        source = self.assemble(options={"host_bytes": 0})["expert_source"]
        source.add(1, 4, *expert_weights(1, 4))
        with open(source.path, "r+b") as file:
            first = file.read(1)
            file.seek(0)
            file.write(bytes([first[0] ^ 0xFF]))
        with self.assertRaisesRegex(ValueError, "changed"):
            source.matrices(1, 4)
        path = source.path
        source.close()
        self.assertFalse(os.path.exists(path))
