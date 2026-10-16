"""Live decisions using actual paced transfers and generation-leased host tiers."""
from concurrent.futures import Future
from dataclasses import replace
import unittest
from test_techniques import load


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.p = load('checkpoint-storage', 'planning.py')
        self.t = load('checkpoint-storage', 'tiers.py')
        self.x = load('checkpoint-storage', 'transfers.py')
        self.m = load('checkpoint-storage', 'prefetch.py')
        self.now = 10.0
        self.c = self.m.PrefetchController(planning=self.p, clock=lambda: self.now, max_jobs=2)
        self.slow, self.fast = self.t.MemoryTier(1024), self.t.MemoryTier(1024)
        self.cost = self.p.PrefillCost(10, 0, 1)
        self.identity = dict(fingerprint='weights+conditioning', codec='opaque.v1', components={'kv', 'recurrence'})
        self.moves, self.pending = [], []
        self.proved = False

    def source(self, name='long', length=8):
        return self.p.PrefixSource(name, tuple(range(length)), **self.identity, ready=False)

    def start(self, key='request', source=None, *, delayed=False, deadline=20, remaining=1, corrupt=False, ambiguous=False):
        source = source or self.source()
        self.slow.put(key, 'complete-bundle', b'abcdefgh')
        def read(move, window):
            receipt = window.record(Future())
            if ambiguous:
                self.pending.append((move, window, receipt))
                raise RuntimeError('unknown enqueue')
            if delayed:
                self.pending.append((move, window, receipt))
            else:
                window.buffer[:] = b'X' * window.nbytes if corrupt else move.lease.data[window.offset:window.offset+window.nbytes]
                receipt.set_result(window.nbytes)
            return receipt
        def prepare():
            move = self.t.TierMove(self.slow, self.fast, key, 'complete-bundle',
                                   self.x.TransferLimits(4, 8, 2, 8), transport=self.x, read=read)
            self.moves.append(move)
            return self.m.tier_prefetch(source, move, lambda: remaining, settle=lambda _: self.proved)
        self.c.start(key, prepare, deadline=deadline)

    def decide(self, key='request', **kw):
        return self.c.decide(key, range(10), self.cost, **self.identity, **kw)

    def complete_readers(self):
        for move, window, receipt in self.pending:
            if not receipt.done():
                window.buffer[:] = move.lease.data[window.offset:window.offset+window.nbytes]
                receipt.set_result(window.nbytes)

    def tearDown(self):
        self.complete_readers()
        for move in self.moves:
            if not move._closed:
                move.discard(lambda _: True)  # The fixtures created no untracked physical work.

    def test_live_wait_then_complete_lease_and_whole_publication(self):
        self.start()
        self.assertEqual(self.decide().action, 'wait')
        self.assertIsNone(self.fast.generation('request'))
        for _ in range(10):
            self.assertLessEqual(self.c.pump(max_bytes=4, max_submissions=1), 4)
        decision = self.decide()
        self.assertEqual((decision.action, decision.source.prefix), ('reuse', tuple(range(8))))
        self.assertEqual(decision.lease.data, b'abcdefgh')
        self.assertEqual(self.c.retained, ())
        self.fast.evict('request', self.fast.generation('request'))
        self.assertEqual(self.fast.used, 8)  # Request owner outlives eviction.
        decision.lease.close()
        self.assertEqual(self.fast.used, 0)

    def test_deadline_falls_back_to_shorter_prefix_retains_pending_reader(self):
        self.start(delayed=True, deadline=11)
        self.c.pump(max_bytes=4)
        self.fast.put('short', 'complete-bundle', b'short')
        short = self.m.tier_prefix(replace(self.source('short', 4), ready=True), self.fast, 'short', 'complete-bundle')
        self.now = 11
        result = self.decide(alternatives=(short,))
        self.assertEqual((result.action, len(result.source.prefix)), ('reuse', 4))
        self.assertEqual(self.c.retained, ('request',))
        self.assertEqual(self.fast.used, 13)
        self.complete_readers()
        self.c.collect()
        self.assertEqual(self.c.retained, ())
        self.assertEqual(self.fast.used, 5)
        result.lease.close()

    def test_unknown_or_expensive_estimate_recomputes(self):
        for estimate in (None, 100, float('nan')):
            self.start(remaining=estimate)
            self.assertEqual(self.decide().action, 'recompute')
            self.assertFalse(self.c.retained)

    def test_aggregate_budget_round_robin_and_capacity_before_prepare(self):
        self.start('a', delayed=True)
        self.start('b', delayed=True)
        called = []
        with self.assertRaises(ValueError):
            self.c.start('c', lambda: called.append(True), deadline=20)
        self.assertEqual(called, [])
        self.assertEqual(self.c.pump(max_bytes=3, max_submissions=2), 0)
        self.c.pump(max_bytes=4, max_submissions=1)
        self.c.pump(max_bytes=4, max_submissions=1)
        self.assertEqual([move.transfer.read_bytes for move in self.moves], [4, 4])
        self.c.cancel('a')
        self.assertIn('a', self.c.retained)
        self.assertEqual(self.moves[1].transfer.state, 'active')
        self.complete_readers()
        self.c.collect()
        self.assertEqual(self.c.retained, ('b',))

    def test_source_replacement_never_publishes_old_generation(self):
        self.start(delayed=True)
        self.c.pump(max_bytes=4)
        self.slow.put('request', 'complete-bundle', b'new')
        self.assertEqual(self.decide().action, 'recompute')
        self.assertIsNone(self.fast.generation('request'))
        self.assertEqual(self.slow.used, 11)
        self.complete_readers()
        self.c.collect()
        self.assertEqual(self.slow.used, 3)

    def test_stale_ready_alternative_falls_back_without_wrong_lease(self):
        self.fast.put('short', 'complete-bundle', b'old')
        short = self.m.tier_prefix(replace(self.source('short', 4), ready=True), self.fast, 'short', 'complete-bundle')
        self.fast.put('short', 'complete-bundle', b'new')
        self.assertEqual(self.decide(alternatives=(short,)).action, 'recompute')

    def test_corruption_and_destination_race_never_publish(self):
        self.start(corrupt=True)
        for _ in range(10):
            self.c.pump(max_bytes=8, max_submissions=2)
        self.assertEqual(self.decide().action, 'recompute')
        self.assertIsNone(self.fast.generation('request'))
        self.start()
        for _ in range(10):
            self.c.pump(max_bytes=8, max_submissions=2)
        self.fast.put('request', 'complete-bundle', b'other')
        self.assertEqual(self.decide().action, 'recompute')
        lease = self.fast.acquire('request', 'complete-bundle')
        self.assertEqual(lease.data, b'other')
        lease.close()

    def test_ambiguous_submission_keeps_slot_until_explicit_proof(self):
        self.start(ambiguous=True)
        self.c.pump(max_bytes=4)
        self.complete_readers()
        self.c.collect()
        self.assertEqual(self.c.retained, ('request',))
        self.assertEqual(self.fast.used, 8)
        # Only the native adapter may establish this proof and retry discard.
        self.proved = True
        self.c.collect()
        self.assertFalse(self.c.retained)

    def test_pump_expiry_stops_submission_before_decision(self):
        self.start(delayed=True, deadline=11)
        self.c.pump(max_bytes=4)
        self.now = 11
        self.assertEqual(self.c.pump(max_bytes=16, max_submissions=4), 0)
        self.assertEqual(self.moves[0].transfer.read_bytes, 4)
        self.assertIn('request', self.c.retained)
        self.complete_readers()
        self.c.collect()
        self.assertFalse(self.c.retained)

    def test_incompatible_siblings_do_not_wait_or_publish(self):
        self.start(source=replace(self.source(), components=frozenset({'kv'})))
        self.assertEqual(self.decide().action, 'recompute')
        self.assertIsNone(self.fast.generation('request'))


if __name__ == '__main__':
    unittest.main()
