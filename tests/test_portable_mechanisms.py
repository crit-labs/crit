"""Portable metadata, state coordination, replay policy and real host tier copies."""
from concurrent.futures import Future
import random
import unittest

from test_techniques import load


class CoordinationTests(unittest.TestCase):
    def test_heterogeneous_layout_and_capacity(self):
        m = load("checkpoint-storage", "coordination.py")
        regions, used = m.plan_state_layout((m.StateLayout("kv", 33, 3, 16),
                                             m.StateLayout("recurrence", 7, 2, 8)), 160)
        self.assertEqual([(r.offset, r.stride, r.nbytes) for r in regions], [(0, 48, 144), (144, 8, 16)])
        self.assertEqual(used, 160)
        with self.assertRaises(ValueError):
            m.plan_state_layout((m.StateLayout("x", 33, 3, 16),), 143)
        with self.assertRaises(ValueError):
            m.plan_state_layout((m.StateLayout("x", 1 << 62, 8),), (1 << 63) - 1)

    def test_non_downward_closed_prefixes_need_reconciliation(self):
        m = load("checkpoint-storage", "coordination.py")
        components = (m.PrefixFrontier("kv", "weights", 100, 4),
                      m.PrefixFrontier("recurrent", "weights", 100, checkpoints=(16, 30, 70)),
                      m.PrefixFrontier("auxiliary", "weights", 100, checkpoints=(16, 60, 90)))
        self.assertEqual(m.common_prefix(components, 100, identity="weights"), 16)
        with self.assertRaises(ValueError):
            m.common_prefix(components, 100, identity="different weights")
        # Coprime dense strides must not cause a walk over 10**18 tokens.
        dense = (m.PrefixFrontier("a", "x", 10**18, 1000000007),
                 m.PrefixFrontier("b", "x", 10**18, 1000000009))
        self.assertEqual(m.common_prefix(dense, 10**18, identity="x"), 0)

    def test_random_frontiers_against_set_intersection(self):
        m, rng = load("checkpoint-storage", "coordination.py"), random.Random(405)
        for _ in range(400):
            components, allowed = [], set(range(101))
            limit = rng.randrange(101)
            for i in range(rng.randrange(1, 7)):
                maximum, quantum = rng.randrange(101), rng.randrange(1, 9)
                values = set(range(0, maximum + 1, quantum))
                checkpoints = None
                if rng.randrange(2):
                    values = {v for v in values if v == 0 or rng.randrange(2)}
                    checkpoints = tuple(sorted(values - {0}))
                components.append(m.PrefixFrontier(str(i), "x", maximum, quantum, checkpoints))
                allowed &= values
            self.assertEqual(m.common_prefix(components, limit, identity="x"), max(v for v in allowed if v <= limit))


class TierTests(unittest.TestCase):
    def setUp(self):
        self.m = load("checkpoint-storage", "tiers.py")
        self.t = load("checkpoint-storage", "transfers.py")
        self.limits = self.t.TransferLimits(3, 6, 2, 100)

    def finish(self, move):
        for _ in range(100):
            move.transfer.pump(max_bytes=6, max_submissions=2)
            if move.transfer.done():
                return
        self.fail("host move did not settle")

    def test_real_copy_publication_and_retained_source_reader(self):
        source, destination = self.m.MemoryTier(30), self.m.MemoryTier(30)
        source.put("prefix", "model", b"abcdefgh")
        reader = source.acquire("prefix", "model")
        move = self.m.TierMove(source, destination, "prefix", "model", self.limits, transport=self.t)
        self.assertIsNone(destination.generation("prefix"))
        self.finish(move)
        generation = move.publish(evict_source=True)
        result = destination.acquire("prefix", "model")
        self.assertEqual(result.data, b"abcdefgh")
        self.assertIsNone(source.generation("prefix"))
        self.assertEqual(source.used, 8)
        reader.close()
        self.assertEqual(source.used, 0)
        result.close()
        destination.evict("prefix", generation)
        self.assertEqual(destination.used, 0)
        with self.assertRaises(ValueError):
            move.publish()

    def test_source_and_destination_generation_races(self):
        for changed in ("source", "destination"):
            source, dest = self.m.MemoryTier(40), self.m.MemoryTier(40)
            source.put("k", "i", b"old")
            move = self.m.TierMove(source, dest, "k", "i", self.limits, transport=self.t)
            self.finish(move)
            (source if changed == "source" else dest).put("k", "i", b"new")
            with self.assertRaisesRegex(ValueError, "changed"):
                move.publish()
            move.discard()
            self.assertEqual(source.used, 3)
            self.assertEqual(dest.used, 3 if changed == "destination" else 0)

    def test_pending_cancel_keeps_both_tiers_charged(self):
        source, dest = self.m.MemoryTier(10), self.m.MemoryTier(10)
        generation = source.put("k", "i", b"abc")
        pending = Future()
        def write(move, window):
            window.record(pending)
            move._private[window.offset:window.offset + window.nbytes] = window.buffer
            return pending
        move = self.m.TierMove(source, dest, "k", "i", self.limits, transport=self.t, write=write)
        move.transfer.pump(max_bytes=3)
        move.transfer.pump(max_bytes=3)
        source.evict("k", generation)
        with self.assertRaises(self.t.PendingTransfer):
            move.discard()
        self.assertEqual((source.used, dest.used), (3, 3))
        pending.set_result(3)
        move.discard()
        self.assertEqual((source.used, dest.used), (0, 0))

    def test_unknown_submission_and_corruption_never_publish(self):
        for fault in ("unknown", "corrupt"):
            source, dest = self.m.MemoryTier(10), self.m.MemoryTier(10)
            source.put("k", "i", b"abc")
            def write(move, window):
                receipt = window.record(Future())
                if fault == "unknown":
                    raise RuntimeError("submission outcome unknown")
                move._private[:] = b"bad"
                receipt.set_result(window.nbytes)
                return receipt
            move = self.m.TierMove(source, dest, "k", "i", self.limits, transport=self.t, write=write)
            if fault == "unknown":
                with self.assertRaises(self.t.TransferFailure):
                    self.finish(move)
                with self.assertRaises(self.t.PendingTransfer):
                    move.discard()
                self.assertEqual(dest.used, 3)
                move.discard(lambda window: True)
            else:
                self.finish(move)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    move.publish()
                move.discard()
            self.assertIsNone(dest.generation("k"))
            self.assertEqual(dest.used, 0)

    def test_capacity_and_empty_state(self):
        source, dest = self.m.MemoryTier(20), self.m.MemoryTier(5)
        source.put("k", "i", b"abc")
        move = self.m.TierMove(source, dest, "k", "i", self.limits, transport=self.t)
        self.finish(move)
        with self.assertRaisesRegex(ValueError, "capacity"):
            move.publish()  # Immutable conversion needs another three bytes.
        self.assertIsNone(dest.generation("k"))
        move.discard()
        source.put("empty", "i", b"")
        move = self.m.TierMove(source, dest, "empty", "i", self.limits, transport=self.t)
        self.finish(move)
        move.publish()
        lease = dest.acquire("empty", "i")
        self.assertEqual(lease.data, b"")
        lease.close()


if __name__ == "__main__":
    unittest.main()
