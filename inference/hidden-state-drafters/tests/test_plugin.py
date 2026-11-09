from crit_plugins.testing import PluginTestCase

FACTS = ["draft:head", "draft:head-device", "taps:hidden-states"]


class Head:
    """A toy drafter over a 4-token draft vocabulary mapped to target ids 10..13 by d2t; sure of its first draft
    only."""
    d2t = [10, 10, 10, 10]

    def __call__(self, paths, rows=None):
        return [[5.0, 0.0, 0.0, 0.0] if not p else [0.1, 0.0, 0.2, 0.0] for p in paths]


class HiddenStateDraftersTests(PluginTestCase):
    def test_it_has_the_drafter_interface(self):
        self.assertImplements(self.assemble(facts=FACTS))

    def test_drafts_are_mapped_to_target_ids(self):
        drafter = self.assemble(facts=FACTS)["drafter"]
        drafter.start(1, [4])
        self.assertEqual(drafter.propose(1, 3, Head())["tokens"], [4, 10, 12, 12])

    def test_the_cut_learns_to_drop_unsure_drafts(self):
        drafter = self.assemble(options={"cut_target": 0.8}, facts=FACTS)["drafter"]
        drafter.start(1, [4])
        for _ in range(80):   # the sure draft is always accepted, the unsure ones never
            proposal = drafter.propose(1, 3, Head())
            drafter.observe(1, [10, 11])
        self.assertEqual(proposal["tokens"][1], 10)
        self.assertEqual(drafter.propose(1, 3, Head())["tokens"], [11, 10, -1, -1])
