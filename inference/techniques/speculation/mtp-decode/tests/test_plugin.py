from crit_plugins.testing import PluginTestCase


class MTPDecodeTests(PluginTestCase):
    def test_it_has_the_drafter_interface(self):
        self.assertImplements()

    def test_it_copies_the_continuation_of_the_last_tokens(self):
        drafter = self.assemble(options={"window": 2})["drafter"]
        drafter.start(1, [7, 1, 2, 3, 4, 5, 9, 1, 2])
        proposal = drafter.propose(1, 3)
        self.assertEqual(proposal["tokens"], [2, 3, 4, 5])
        self.assertEqual(proposal["parents"], [-1, 0, 1, 2])
        drafter.observe(1, [3, 8])
        self.assertEqual(drafter.propose(1, 2)["tokens"], [8])   # no earlier continuation of (3, 8)
