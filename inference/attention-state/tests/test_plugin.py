from crit_plugins.slots import CASES
from crit_plugins.testing import PluginTestCase


class AttentionStateTests(PluginTestCase):
    def test_it_has_the_kv_cache_interface(self):
        self.assertImplements()

    def test_commit_keeps_the_accepted_prefix_and_a_fork_shares_it(self):
        out = CASES["kv_cache"](self.assemble())
        self.assertEqual(len(out["pending"]), 5)
        self.assertEqual([row[0] for row in out["committed"]], [0, 1, 2, 3])
        self.assertEqual(len(out["fork"]), 6)
        self.assertEqual(out["fork"][:4], out["parent"])
        self.assertEqual(len(out["parent"]), 4)

    def test_capacity_is_kept(self):
        kv = self.assemble()["kv_cache"]
        kv.open(1, 2, 2, 2)
        kv.stage(1, [[0.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [2.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "capacity"):
            kv.stage(1, [[0.0, 0.0]], [[0.0, 0.0]])
