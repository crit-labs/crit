from crit_plugins.testing import PluginTestCase

FACTS = ["draft:head", "draft:head-device", "draft:head-rows"]


def head(paths, rows=None):
    """A toy draft model over 12 tokens: after a path of n tokens the best is (n * 5 + 2) % 12."""
    full = [[1.0 if t == (len(p) * 5 + 2) % 12 else 0.0 for t in range(12)] for p in paths]
    return full if rows is None else [[r[t] for t in rows] for r in full]


class DraftVocabularySubsetTests(PluginTestCase):
    def test_it_has_the_drafter_interface(self):
        self.assertImplements(self.assemble(facts=FACTS))

    def test_with_the_whole_vocabulary_it_drafts_the_greedy_chain(self):
        subset = self.assemble(options={"size": 12}, facts=FACTS)["drafter"]
        chain = self.reference(facts=FACTS)["drafter"]
        for drafter in (subset, chain):
            drafter.start(1, [3])
        self.assertEqual(subset.propose(1, 4, head), chain.propose(1, 4, head))

    def test_it_drafts_only_inside_its_subset_and_learns_emitted_tokens(self):
        drafter = self.assemble(options={"size": 4, "refresh": 2}, facts=FACTS)["drafter"]
        drafter.start(1, [3])
        self.assertTrue(set(drafter.propose(1, 3, head)["tokens"][1:]) <= {0, 1, 2, 3})
        drafter.observe(1, [7, 7])
        drafter.observe(1, [7])
        self.assertIn(7, drafter.propose(1, 3, head)["tokens"][1:])   # (1 * 5 + 2) % 12 = 7 is in the subset now
