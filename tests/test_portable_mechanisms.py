"""Portable metadata, state coordination, replay policy and real host tier copies."""
from concurrent.futures import Future
from dataclasses import replace
import random
import unittest

from test_techniques import load


class MetadataTests(unittest.TestCase):
    def test_incremental_gather_rebind_and_retained_snapshots(self):
        m = load("concurrent-request-execution", "metadata.py")
        table = m.MetadataTable(1000000, 32)
        replica = m.MetadataReplica(table.empty)
        table.set((1, 1), b"abc")
        table.set((2, 1), b"de")
        first = table.snapshot()
        self.assertEqual(first.diff(replica.snapshot).payload_bytes, 5)
        replica.apply(first.diff(replica.snapshot))
        held = replica.snapshot
        table.set((1, 1), b"abcd")
        second = table.snapshot(((2, 1), (1, 1)))
        delta = second.diff(replica.snapshot)
        self.assertEqual(delta.payload_bytes, 4)
        self.assertEqual(replica.apply(delta), second)
        self.assertEqual(held.rows[0].payload, b"abc")
        with self.assertRaises(ValueError):
            replica.apply(delta)
        table.release((1, 1))
        self.assertEqual(table.set((3, 2), b"new"), 0)
        third = table.snapshot()
        self.assertEqual(replica.apply(third.diff(replica.snapshot)), third)
        self.assertEqual(held.rows[0].key, (1, 1))
        same = table.snapshot()
        self.assertEqual(same.diff(replica.snapshot).payload_bytes, 0)
        replica.apply(same.diff(replica.snapshot))

    def test_random_patches_match_full_rebuild(self):
        m, rng = load("concurrent-request-execution", "metadata.py"), random.Random(144)
        table = m.MetadataTable(12, 20)
        replica = m.MetadataReplica(table.empty)
        expected = {}
        for _ in range(500):
            key = (rng.randrange(1, 13), 1)
            if key in expected and rng.randrange(4) == 0:
                table.release(key)
                del expected[key]
            else:
                value = rng.randbytes(rng.randrange(20))
                table.set(key, value)
                expected[key] = value
            keys = list(expected)
            rng.shuffle(keys)
            wanted = table.snapshot(keys)
            actual = replica.apply(wanted.diff(replica.snapshot))
            self.assertEqual([(r.key, r.payload) for r in actual.rows], [(k, expected[k]) for k in keys])

    def test_invalid_patches_are_atomic(self):
        m = load("concurrent-request-execution", "metadata.py")
        table = m.MetadataTable(1, 3)
        replica = m.MetadataReplica(table.empty)
        table.set((1, 1), b"one")
        delta = table.snapshot().diff(replica.snapshot)
        for bad in (replace(delta, order=(0, 0)), replace(delta, changed=(replace(delta.changed[0], payload=b"long"),)),
                    replace(delta, changed=(replace(delta.changed[0], slot=9),))):
            with self.assertRaises(ValueError):
                replica.apply(bad)
            self.assertIs(replica.snapshot, table.empty)
        with self.assertRaises(ValueError):
            table.set((2, 1), b"two")
        replica.apply(delta)
        bad = m.MetadataDelta(replica.snapshot, 99, (0,),
                              (replace(replica.snapshot.rows[0], version=1),), (0,))
        with self.assertRaises(ValueError):
            replica.apply(bad)
        foreign = m.MetadataTable(1, 3)
        with self.assertRaises(ValueError):
            foreign.snapshot().diff(replica.snapshot)


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


class ReplayTests(unittest.TestCase):
    def test_amortization_memory_padding_and_generation(self):
        m = load("recorded-selection", "replay.py")
        shape = m.ExecutionShape("decode", 3, 2, ("repr", "backend", 7, (1, 2)))
        eager = m.ReplayOffer("eager", "eager", shape, 100, workspace_bytes=2)
        full = m.ReplayOffer("full", "full", replace(shape, tokens=4), 20,
                             workspace_bytes=4, resident_bytes=8, setup_ns=500, setup_peak_bytes=20)
        partial = m.ReplayOffer("partial", "partial", shape, 50, workspace_bytes=4, ready=True)
        def select(offers, **kw):
            return m.select_replay(shape, offers, modes={"eager", "full", "partial"}, **kw)
        self.assertEqual(select((eager, full), free_bytes=20, allow_padding=True).offer, eager)
        self.assertEqual(select((eager, full), free_bytes=20, expected_uses=10, allow_padding=True).offer, full)
        self.assertEqual(select((eager, full, partial), free_bytes=4, expected_uses=10, allow_padding=True).offer, partial)
        self.assertEqual(select((eager, full), free_bytes=20, expected_uses=10).offer, eager)
        stale = replace(partial, shape=replace(shape, identity=("repr", "backend", 8, (1, 2))))
        choice = select((eager, stale), free_bytes=100)
        self.assertEqual(choice.offer, eager)
        self.assertIn("identity", choice.rejected[0][1])
        self.assertEqual(select((replace(full, ready=True),), free_bytes=4, allow_padding=True).offer.mode, "full")
        with self.assertRaises(ValueError):
            select((eager,), free_bytes=1)
        with self.assertRaises(ValueError):
            m.ExecutionShape("decode", 1, 1, (object(),))


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
