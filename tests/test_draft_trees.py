"""Host fixtures for draft-trees' reference: ancestor-closed kept sets, masks and depths, chains walking as greedy
speculation, emitted tokens that are always the target's, and compaction that keeps exactly the accepted rows."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    name = "crit_test_draft_trees_tree"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "inference" / "draft-trees" / "code" / "tree.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def random_drafter(seed, vocab=12):
    def rows(paths):
        out = []
        for path in paths:
            rng = random.Random(hash((seed,) + tuple(path)))
            out.append([rng.gauss(0, 2) for _ in range(vocab)])
        return out
    return rows


class DraftTreeTests(unittest.TestCase):
    def test_trees_are_ancestor_closed_with_path_masks(self):
        m = load()
        for seed in range(30):
            candidates, tree = m.draft(random_drafter(seed), root=5, depth=4, branch=3, beam=3, nodes=1 + seed % 20)
            self.assertEqual(tree.tokens[0], 5)
            for i in range(1, tree.nodes):
                p = tree.parents[i]
                self.assertLess(p, i)
                self.assertEqual(tree.masks[i], tree.masks[p] | (1 << i))
                self.assertEqual(tree.depths[i], bin(tree.masks[i]).count("1") - 1)
            # every kept candidate's parent is kept (scores never grow along a path)
            kept = m.rank(candidates, range(len(candidates)), tree.nodes - 1)
            self.assertTrue(all(candidates[k].parent in kept or candidates[k].parent == -1 for k in kept))

    def test_walk_emits_only_target_tokens(self):
        m = load()
        rng = random.Random(4)
        for seed in range(50):
            _, tree = m.draft(random_drafter(seed), root=0, depth=3, branch=2, beam=2, nodes=9)
            predictions = [rng.randrange(12) for _ in range(tree.nodes)]
            emitted, path = m.accept(tree, predictions, remaining=6, stops=(11,))
            self.assertEqual(path[0], 0)
            # each emitted token is the prediction of the node before it on the path
            for i, token in enumerate(emitted):
                self.assertEqual(token, predictions[path[i]])
            for a, b in zip(path, path[1:]):
                self.assertEqual(tree.parents[b], a)
            self.assertLessEqual(len(emitted), 6)
            moves = m.compaction(path, len(emitted))
            rows = m.apply(list(range(tree.nodes)), moves)
            self.assertEqual(rows[:len(emitted)], path[:len(emitted)])

    def test_chains_walk_as_greedy_speculation(self):
        m = load()
        line = m.chain(9, [1, 2, 3, 4])
        self.assertEqual(line.masks, (1, 3, 7, 15, 31))
        self.assertEqual(m.accept(line, [1, 2, 3, 4, 8], remaining=9), ([1, 2, 3, 4, 8], [0, 1, 2, 3, 4]))
        self.assertEqual(m.accept(line, [1, 7, 0, 0, 0], remaining=9), ([1, 7], [0, 1]))
        self.assertEqual(m.accept(line, [1, 2, 3, 4, 8], remaining=2), ([1, 2], [0, 1, 2]))
        self.assertEqual(m.accept(line, [1, 2, 3, 4, 8], remaining=9, stops=(2,)), ([1, 2], [0, 1, 2]))
        with self.assertRaises(ValueError):
            m.accept(line, [1, 2], remaining=3)
        with self.assertRaises(ValueError):
            m.draft(random_drafter(0), root=0, depth=2, branch=2, beam=2, nodes=65)

    def test_draft_ranks_measure_what_a_wider_tree_would_accept(self):
        m = load()
        drafter, vocab, K, ranks = random_drafter(5), 12, 4, 3
        stats = m.DraftRanks(K, ranks)
        rng = random.Random(8)
        accepted_at, compared_at, cut = [0] * K, [0] * K, 0
        for trial in range(300):
            root = trial % vocab
            rows, path, drafts = [], [], []   # drafter paths are the tokens after the root
            for _ in range(K):   # a chain of the drafter's first choices
                rows.append(drafter([path])[0])
                drafts.append(m.expand(rows[-1], 1)[0][0])
                path.append(drafts[-1])
            # The target takes the draft or a token of rank 1 .. 5 in the drafter's row.
            predictions = [m.expand(r, 6)[0 if rng.random() < 0.6 else rng.randrange(1, 6)][0] for r in rows] + [0]
            # Every third round has a smaller output allowance, so some end on the allowance with nothing rejected.
            emitted, walk = m.accept(m.chain(root, drafts), predictions, remaining=K + 1 if trial % 3 else 1 + trial % K)
            matched = len(walk) - 1
            for d in range(matched):
                accepted_at[d] += 1
                compared_at[d] += 1
            if len(emitted) > matched and matched < K:
                compared_at[matched] += 1   # the correction's position was compared
            cut += len(emitted) == matched
            stats.record(rows, predictions, matched, len(emitted))
            # At position 0 a one-level tree of width w holds the target's token exactly when its rank is below w.
            for width in range(1, 6):
                _, tree = m.draft(drafter, root=root, depth=1, branch=width, beam=1, nodes=width + 1)
                held = m.accept(tree, [predictions[0]] + [0] * width, remaining=2)[1] != [0]
                self.assertEqual(held, m.draft_rank(rows[0], predictions[0]) < width)
        self.assertEqual([stats.counts[d][0] for d in range(K)], accepted_at)
        self.assertEqual([stats.reached(d) for d in range(K)], compared_at)
        self.assertGreater(cut, 0)   # rounds cut off by their allowance occurred and compared nothing past their drafts
        for d in range(K):
            shares = [stats.covered(d, w) for w in range(1, ranks + 2)]
            self.assertEqual(shares, sorted(shares))
            self.assertEqual(shares[-1], 1.0 if stats.reached(d) else 0.0)
        self.assertGreater(stats.counts[0][ranks], 0)   # ranks >= 3 land in the last column
        with self.assertRaises(ValueError):
            m.DraftRanks(0, 1)
        with self.assertRaises(ValueError):
            stats.record(rows, predictions, 1, 3)   # at most one token past the accepted drafts


if __name__ == "__main__":
    unittest.main()
