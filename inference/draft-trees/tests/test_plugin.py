from crit_plugins.testing import PluginTestCase

FACTS = ["draft:head", "draft:head-device"]


def head(paths, rows=None):
    """A toy draft model whose rows depend on the path: best token (sum + 1) % 9, then (sum + 4) % 9."""
    out = []
    for p in paths:
        s = sum(p)
        out.append([2.0 if t == (s + 1) % 9 else 1.0 if t == (s + 4) % 9 else 0.0 for t in range(9)])
    return out


CHAIN = {"branch": 1, "beam": 1, "depth": 4}


class DraftTreesTests(PluginTestCase):
    def test_it_has_the_drafter_and_verifier_interfaces(self):
        self.assertImplements(self.assemble(facts=FACTS))

    def test_one_object_fills_both_slots(self):
        trees = self.assemble(facts=FACTS)
        self.assertIs(trees["drafter"], trees["verifier"])

    def test_with_one_branch_it_drafts_the_greedy_chain(self):
        trees = self.assemble(options=CHAIN, facts=FACTS)["drafter"]
        chain = self.reference(facts=FACTS)["drafter"]
        for drafter in (trees, chain):
            drafter.start(3, [8, 2])
        self.assertEqual(trees.propose(3, 4, head), chain.propose(3, 4, head))

    def test_on_chains_its_accept_equals_chain_verifier(self):
        self.assertMatchesReference(facts=FACTS)

    def test_with_one_branch_a_round_equals_the_separate_pair(self):
        rounds = []
        for assembly in (self.assemble(options=CHAIN, facts=FACTS), self.reference(facts=FACTS)):
            drafter, verifier, out = assembly["drafter"], assembly["verifier"], []
            drafter.start(0, [5])
            for predictions in ([1, 2, 3, 4, 5], [6, 0, 0, 0, 0]):
                proposal = drafter.propose(0, 4, head)
                result = verifier.accept(proposal, predictions, 10)
                drafter.observe(0, result["tokens"])
                out.append((proposal["tokens"], result["tokens"], result["accepted"]))
            rounds.append(out)
        self.assertEqual(rounds[0], rounds[1])

    def test_a_tree_keeps_every_nodes_parent_and_the_best_child_first(self):
        drafter = self.assemble(facts=FACTS)["drafter"]
        drafter.start(1, [5])
        proposal = drafter.propose(1, 6, head)
        tokens, parents = proposal["tokens"], proposal["parents"]
        self.assertEqual(len(tokens), 7)
        self.assertEqual(parents[0], -1)
        self.assertTrue(all(0 <= parents[i] < i for i in range(1, 7)))
        self.assertEqual(tokens[1], 1)   # the root's best child: (0 + 1) % 9
        self.assertGreater(sum(1 for p in parents if p == 0), 1)   # more than one branch at the root

    def test_it_walks_the_branch_the_target_predicts(self):
        verifier = self.assemble(facts=FACTS)["verifier"]
        # root 9 with children 4 (node 1) and 6 (node 2); node 2 has child 7 (node 3)
        proposal = {"tokens": [9, 4, 6, 7], "parents": [-1, 0, 0, 2]}
        self.assertEqual(verifier.accept(proposal, [6, 1, 7, 8], 10), {"tokens": [6, 7, 8], "accepted": 2,
                                                                        "path": [0, 2, 3]})
        self.assertEqual(verifier.accept(proposal, [5, 1, 7, 8], 10)["tokens"], [5])

    def test_its_own_tree_round_emits_the_branch_the_target_predicts(self):
        trees = self.assemble(facts=FACTS)["drafter"]
        trees.start(1, [5])
        proposal = trees.propose(1, 6, head)
        second = next(i for i in range(2, len(proposal["tokens"])) if proposal["parents"][i] == 0)
        predictions = [proposal["tokens"][second]] + [0] * (len(proposal["tokens"]) - 1)
        result = trees.accept(proposal, predictions, 10)
        self.assertEqual(result["path"][:2], [0, second])
        self.assertEqual(result["tokens"][0], proposal["tokens"][second])
