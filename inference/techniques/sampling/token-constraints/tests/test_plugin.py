import math

from crit_plugins.testing import PluginTestCase

ROW = [2.0, 1.0, 2.0, -1.0, 0.5, 1.5]


class TokenConstraintsTests(PluginTestCase):
    def test_it_has_the_sampler_interface(self):
        self.assertImplements()

    def test_it_equals_core_sampler_on_the_slot_cases(self):
        self.assertMatchesReference(atol=1e-12, rtol=1e-9)

    def test_the_law_has_no_mass_outside_the_allowed_set(self):
        sampler = self.assemble()["sampler"]
        law = dict(sampler.law(ROW, {"temperature": 0.9, "allowed": [1, 3, 5]}))
        self.assertEqual(set(law), {1, 3, 5})
        self.assertAlmostEqual(math.fsum(law.values()), 1.0)
        self.assertEqual(sampler.select([ROW], {"temperature": 0.0, "allowed": [3, 4]}), [4])

    def test_a_rejected_draft_is_repaired_inside_the_language(self):
        sampler = self.assemble()["sampler"]
        rule = {"temperature": 1.0, "seed": 2, "allowed": [0, 2, 5]}
        repaired = set()
        for position in range(60):
            # A draft law that puts most mass on token 5: often rejected, and the repair stays allowed.
            out = sampler.predictions([ROW, ROW], [5], dict(rule, position=position), [{5: 0.9, 0: 0.05, 2: 0.05}])
            repaired.add(out[0])
        self.assertTrue(repaired <= {0, 2, 5})
        self.assertGreater(len(repaired), 1)
