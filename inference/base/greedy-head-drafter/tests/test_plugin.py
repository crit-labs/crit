from crit_plugins.testing import PluginTestCase


def head(paths, rows=None):
    """A toy draft model: after a path of n tokens it prefers token (n * 3 + 1) % 7."""
    return [[1.0 if t == (len(p) * 3 + 1) % 7 else 0.0 for t in range(7)] for p in paths]


class GreedyHeadDrafterTests(PluginTestCase):
    def test_it_has_the_drafter_interface(self):
        self.assertImplements(self.assemble(facts=["draft:head", "draft:head-device"]))

    def test_it_drafts_a_greedy_chain_after_the_pending_token(self):
        drafter = self.assemble(facts=["draft:head", "draft:head-device"])["drafter"]
        drafter.start(1, [5, 6])
        self.assertEqual(drafter.propose(1, 3, head), {"tokens": [6, 1, 4, 0], "parents": [-1, 0, 1, 2], "laws": None})
        drafter.observe(1, [1, 2])
        self.assertEqual(drafter.propose(1, 1, head)["tokens"], [2, 1])
