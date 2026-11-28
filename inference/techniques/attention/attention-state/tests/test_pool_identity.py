"""A step's cached metadata is keyed by its pool's own identity, never by id(pool): a new pool Python allocates at a
freed pool's address, with the same step count, reads its own metadata, not the old pool's."""
import sys
from unittest import mock

from crit_plugins.testing import PluginTestCase


class Pool:
    def __init__(self, steps):
        self.steps = steps


class PoolIdentityTests(PluginTestCase):
    def module(self):
        kv = self.assemble()["kv_cache"]
        return sys.modules[type(kv).__module__].device_attention

    def test_a_new_pool_at_a_freed_pools_address_reads_its_own_step_metadata(self):
        module = self.module()
        reader = module.DeviceAttention.__new__(module.DeviceAttention)
        reader._steps = {}
        made = []
        windows = lambda pool, rows, device: made.append(pool.name) or ("metadata of", pool.name)
        batch = mock.Mock(row=lambda r: (r, 0, 1, 0))
        with mock.patch.object(module, "_windows_of", windows):
            old = Pool(steps=3)
            old.name = "old"
            address = id(old)
            self.assertEqual(reader._step_windows(old, batch, [0], "cpu")[0], ("metadata of", "old"))
            del old
            new = None
            for _ in range(10000):   # CPython hands a freed object's address to the next object of its size
                candidate = Pool(steps=3)
                if id(candidate) == address:
                    new = candidate
                    break
            if new is None:
                self.skipTest("this interpreter gave no new object the freed pool's address")
            new.name = "new"
            self.assertEqual(reader._step_windows(new, batch, [0], "cpu")[0], ("metadata of", "new"))
        self.assertEqual(made, ["old", "new"], "the new pool's metadata was made, not read from the old pool's")

    def test_each_pool_keeps_one_identity_and_no_two_pools_share_one(self):
        module = self.module()
        first, second = Pool(1), Pool(1)
        self.assertEqual(module.pool_identity(first), module.pool_identity(first))
        self.assertNotEqual(module.pool_identity(first), module.pool_identity(second))
