from crit_plugins.slots import CASES
from crit_plugins.testing import PluginTestCase


class ChainVerifierTests(PluginTestCase):
    def test_it_has_the_verifier_interface(self):
        self.assertImplements()

    def test_the_verifier_cases(self):
        out = CASES["verifier"](self.assemble())
        self.assertEqual(out["all"], {"tokens": [4, 5, 6, 7, 8], "accepted": 4})
        self.assertEqual(out["first"], {"tokens": [3], "accepted": 0})
        self.assertEqual(out["middle"], {"tokens": [4, 5, 2], "accepted": 2})
        self.assertEqual(out["budget"], {"tokens": [4, 5], "accepted": 2})
        self.assertEqual(out["stop"], {"tokens": [4, 5], "accepted": 2})
        self.assertEqual(out["none"], {"tokens": [], "accepted": 0})

    def test_a_cut_draft_ends_the_round_with_the_targets_token(self):
        verifier = self.assemble()["verifier"]
        result = verifier.accept({"tokens": [1, 2, -1, -1], "parents": [-1, 0, 1, 2]}, [2, 3, 9, 9], 10)
        self.assertEqual(result["tokens"], [2, 3])

    def test_a_tree_is_refused(self):
        with self.assertRaisesRegex(ValueError, "draft-trees"):
            self.assemble()["verifier"].accept({"tokens": [1, 2, 3], "parents": [-1, 0, 0]}, [2, 3, 4], 5)
