"""tests/stress.py's load plan: every in-load scenario is scheduled in some load of the tier that names it (a scenario
whose extra no load carries never runs, as reopen_at_64k did not before), and the plan is the one it was before its
code moved into crit.preflight (PLANNED), at the sizes it had."""
import unittest

import stress

# tests/stress.py's plans at crit 77c68c4c, before its code moved into crit.preflight: (tier, spread, arithmetic)
# -> [[name, mode, pressure, sorted extras, plugin tests]] over every fixture; "only" and "two" as their tests ask.
# Since then the first speed load also reopens at the same context ("same": reopen_same_context).
PLANNED = {
    'full/False/False': [
        ["fnext-nvfp4", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short"], True],
        ["fnext-nvfp4", "preserve", "natural", ["cancel", "precheck", "short"], False],
        ["fnext-nvfp4", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "floor", ["short", "smoke"], False],
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "precheck", "short"], False],
        ["fnext-fp4local", "preserve", "natural", ["cancel", "short"], False],
        ["fnext-fp4local", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "floor", ["short", "smoke"], False],
        ["fnext-nvfp4-deep", "speed", "natural", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-fp4local-full", "speed", "natural", ["32k", "cancel", "ladder", "precheck", "short"], False],
    ],
    'full/False/True': [
        ["fnext-nvfp4", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short"], True],
        ["fnext-nvfp4", "preserve", "natural", ["cancel", "precheck", "short"], False],
        ["fnext-nvfp4", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "floor", ["short", "smoke"], False],
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "precheck", "short"], False],
        ["fnext-fp4local", "preserve", "natural", ["cancel", "short"], False],
        ["fnext-fp4local", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "floor", ["short", "smoke"], False],
        ["fnext-nvfp4-deep", "speed", "natural", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-nvfp4-deep", "speed", "tight", ["arithmetic", "cancel", "short", "smoke"], False],
        ["fnext-fp4local-full", "speed", "natural", ["32k", "cancel", "ladder", "precheck", "short"], False],
    ],
    'full/True/False': [
        ["fnext-nvfp4", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short", "spread"], True],
        ["fnext-nvfp4", "preserve", "natural", ["cancel", "precheck", "short"], False],
        ["fnext-nvfp4", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "floor", ["short", "smoke"], False],
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "precheck", "short"], False],
        ["fnext-fp4local", "preserve", "natural", ["cancel", "short"], False],
        ["fnext-fp4local", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "floor", ["short", "smoke"], False],
        ["fnext-nvfp4-deep", "speed", "natural", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-fp4local-full", "speed", "natural", ["32k", "cancel", "ladder", "precheck", "short"], False],
    ],
    'full/True/True': [
        ["fnext-nvfp4", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short", "spread"], True],
        ["fnext-nvfp4", "preserve", "natural", ["cancel", "precheck", "short"], False],
        ["fnext-nvfp4", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "floor", ["short", "smoke"], False],
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "precheck", "short"], False],
        ["fnext-fp4local", "preserve", "natural", ["cancel", "short"], False],
        ["fnext-fp4local", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "floor", ["short", "smoke"], False],
        ["fnext-nvfp4-deep", "speed", "natural", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-nvfp4-deep", "speed", "tight", ["arithmetic", "cancel", "short", "smoke"], False],
        ["fnext-fp4local-full", "speed", "natural", ["32k", "cancel", "ladder", "precheck", "short"], False],
    ],
    'quick/False/False': [
        ["fnext-nvfp4", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short"], True],
        ["fnext-nvfp4", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "short"], False],
    ],
    'quick/False/True': [
        ["fnext-nvfp4", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short"], True],
        ["fnext-nvfp4", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "short"], False],
    ],
    'quick/True/False': [
        ["fnext-nvfp4", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short", "spread"], True],
        ["fnext-nvfp4", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "short"], False],
    ],
    'quick/True/True': [
        ["fnext-nvfp4", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short", "spread"], True],
        ["fnext-nvfp4", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "short"], False],
    ],
    'only': [
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short"], True],
    ],
    'two': [
        ["fnext-fp4local", "speed", "natural", ["32k", "cancel", "drafting", "lower", "precheck", "same", "short"], True],
        ["fnext-fp4local", "preserve", "natural", ["cancel", "precheck", "short"], False],
        ["fnext-fp4local", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-fp4local", "speed", "floor", ["short", "smoke"], False],
        ["fnext-nvfp4", "speed", "natural", ["32k", "cancel", "precheck", "short"], False],
        ["fnext-nvfp4", "preserve", "natural", ["cancel", "short"], False],
        ["fnext-nvfp4", "speed", "tight", ["32k", "cancel", "ladder", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "extreme", ["32k", "cancel", "short", "smoke"], False],
        ["fnext-nvfp4", "speed", "floor", ["short", "smoke"], False],
    ],
}


def planned(loads):
    return [[load[0], load[1], load[2], sorted(load[3]), load[4]] for load in loads]


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
        for extra in ("short", "ladder", "32k", "cancel", "smoke", "drafting", "arithmetic", "lower", "same",
                      "spread", "precheck"):
            self.assertIn(extra, planned)

    def test_the_plan_is_the_one_before_its_code_moved_into_crit_preflight(self):
        for tier in ("full", "quick"):
            for spread in (False, True):
                for arithmetic in (False, True):
                    self.assertEqual(planned(stress.plan_loads(self.cuts, None, tier, spread, arithmetic)),
                                     PLANNED[f"{tier}/{spread}/{arithmetic}"], (tier, spread, arithmetic))
        self.assertEqual(planned(stress.plan_loads(["fnext-fp4local"], ["fnext-fp4local/speed/natural"], "quick")),
                         PLANNED["only"])
        self.assertEqual(planned(stress.plan_loads(["fnext-fp4local", "fnext-nvfp4"], None, "full")), PLANNED["two"])

    def test_the_fixtures_sizes_and_pressures_are_the_ones_they_were(self):
        self.assertEqual({name: stress.sizes(name) for name in self.cuts}, {
            name: stress.Sizes(context=200000 if name.endswith("-full") else 262144, long=32768, lower=65536,
                               cancel=65536, ladder=(24, 96, 192, 400, 1600, 6400)) for name in self.cuts})
        self.assertEqual(stress.PRESSURES, {"natural": None, "tight": 0.58, "extreme": 0.5, "floor": 0.08})
        self.assertEqual((stress.ACTIVATIONS_GIB, stress.FLOOR_REFUSAL), (4.0, "holds no prompt ring"))


if __name__ == "__main__":
    unittest.main()
