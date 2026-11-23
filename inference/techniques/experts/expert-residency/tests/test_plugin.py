from crit_plugins.testing import PluginTestCase


class ResidencyChecks:
    """What every expert_residency must do; the test class gives OPTIONS (holding `slots`)."""

    OPTIONS = {}

    def residency(self, **options):
        return self.assemble(options={**self.OPTIONS, **options})["expert_residency"]

    def test_it_has_the_expert_residency_interface(self):
        self.assertImplements()

    def test_a_plan_answers_each_expert_and_never_holds_more_than_its_slots(self):
        residency = self.residency(slots=3)
        for layer, experts in ((0, [1, 2]), (0, [2, 3, 4]), (1, [0]), (0, [4, 1]), (1, [0, 5, 6, 7])):
            plan = residency.plan(layer, experts)
            self.assertEqual(len(plan), len(experts))
            self.assertTrue(all(isinstance(on, bool) for on in plan))
            resident = residency.resident(layer)
            self.assertTrue({e for e, on in zip(experts, plan) if on} <= resident)
            self.assertLessEqual(len(residency.resident(0)) + len(residency.resident(1)), 3)

    def test_routes_of_one_plan_are_distinct(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.residency().plan(0, [1, 1])


class ExpertResidencyTests(ResidencyChecks, PluginTestCase):
    def test_a_full_plan_leaves_the_rest_off_and_the_next_plan_evicts_the_least_recent(self):
        residency = self.residency(slots=2)
        self.assertEqual(residency.plan(0, [1, 2, 3]), [True, True, False])
        self.assertEqual(residency.plan(0, [2]), [True])
        self.assertEqual(residency.plan(0, [3]), [True])
        self.assertEqual(residency.resident(0), {2, 3})
        self.assertEqual((residency.loaded, residency.released), (3, 1))

    def test_capacity_is_bytes(self):
        residency = self.residency(slots=3, expert_bytes=4096)
        self.assertEqual(residency.cache.capacity, 3 * 4096)
        self.assertEqual(residency.plan(0, [0, 1, 2, 3]), [True, True, True, False])
