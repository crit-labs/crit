from crit_plugins.testing import PluginTestCase


class ResidencyChecks:
    """What every expert_residency must do, under each policy; the test class gives OPTIONS (holding `policy`)."""

    OPTIONS = {}

    def residency(self, **options):
        return self.assemble(options={**self.OPTIONS, **options})["expert_residency"]

    def test_it_has_the_expert_residency_interface(self):
        self.assertImplements(self.assemble(options=self.OPTIONS))

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


class ClockPolicyTests(ResidencyChecks, PluginTestCase):
    OPTIONS = {"policy": "clock"}

    def test_clock_is_the_default_policy(self):
        self.assertIn("stats", vars(self.assemble()["expert_residency"]))

    def test_a_repeated_plan_hits_and_the_clock_evicts_the_unreferenced_slot(self):
        residency = self.residency(slots=2)
        self.assertEqual(residency.plan(0, [1, 2]), [True, True])
        self.assertEqual(residency.plan(0, [1, 2]), [True, True])
        self.assertEqual(residency.stats["hits"], 2)
        self.assertEqual(residency.plan(0, [3]), [True])
        self.assertEqual(len(residency.resident(0)), 2)

    def test_a_zero_quota_bypasses_every_miss(self):
        residency = self.residency(admit_quota=0)
        self.assertEqual(residency.plan(0, [1, 2]), [False, False])
        self.assertEqual(residency.resident(0), set())
        self.assertEqual(residency.stats["bypassed"], 2)

    def test_second_touch_admits_on_the_second_miss(self):
        residency = self.residency(second_touch=True)
        self.assertEqual(residency.plan(1, [7]), [False])
        self.assertEqual(residency.plan(1, [7]), [True])

    def test_more_experts_than_slots_leave_the_rest_off_the_device(self):
        self.assertEqual(self.residency(slots=2).plan(0, [5, 6, 7]), [True, True, False])


class RankedPolicyTests(ResidencyChecks, PluginTestCase):
    OPTIONS = {"policy": "ranked", "layers": 2, "experts": 16}

    def test_misses_are_never_admitted_and_hot_experts_arrive_at_boundaries(self):
        residency = self.residency(slots=2, period=2)
        rounds = []
        for _ in range(6):
            rounds.append(residency.plan(0, [3, 9]))
            residency.plan(1, [1])
        self.assertEqual(rounds[0], [False, False])
        self.assertEqual(rounds[-1], [True, True])
        self.assertEqual(residency.resident(0), {3, 9})
        self.assertGreater(residency.schedule.swapped, 0)
