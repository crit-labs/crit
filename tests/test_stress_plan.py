"""tests/stress.py's load plan: every in-load scenario is scheduled in some load of the tier that names it (a scenario
whose extra no load carries never runs, as reopen_at_64k did not before)."""
import unittest

import stress


def extras_of(loads):
    return set().union(*(load[3] for load in loads)) if loads else set()


class PlanTests(unittest.TestCase):
    cuts = ["fnext-nvfp4", "fnext-fp4local", "fnext-nvfp4-deep", "fnext-fp4local-full"]

    def test_both_tiers_reopen_at_a_lower_context_in_the_first_checkpoints_speed_load(self):
        for tier in ("full", "quick"):
            loads = stress.plan_loads(self.cuts, None, tier)
            lower = [load[:3] for load in loads if "lower" in load[3]]
            self.assertEqual(lower, [("fnext-nvfp4", "speed", "natural")], tier)

    def test_only_selects_the_reopen(self):
        loads = stress.plan_loads(["fnext-fp4local"], ["fnext-fp4local/speed/natural"], "quick")
        self.assertEqual(len(loads), 1)
        self.assertIn("lower", loads[0][3])

    def test_every_extra_in_load_reads_is_planned_in_the_full_tier(self):
        # The extras in_load and load look for: each one must be in a load of the full tier (spread and arithmetic
        # are opt-in flags).
        planned = extras_of(stress.plan_loads(self.cuts, None, "full", spread=True, arithmetic=True))
        for extra in ("short", "ladder", "32k", "cancel", "smoke", "drafting", "arithmetic", "lower", "spread",
                      "precheck"):
            self.assertIn(extra, planned)


if __name__ == "__main__":
    unittest.main()
