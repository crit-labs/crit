"""Host cache-policy/layout fixtures; no device or model qualification."""
from concurrent.futures import Future
from dataclasses import replace
import unittest

from test_techniques import load

p = load("expert-residency", "planning.py")
m = load("expert-residency")


class ExpertCachePlanningTests(unittest.TestCase):
    def component(self, **changes):
        return replace(p.BankComponent("weights", "demo-i8-layout-v1", (4,), 4), **changes)

    def directory(self):
        directory = m.BundleDirectory(2, (m.BundleComponent("weights", "demo", 4),), (object(),))
        keys = [m.ExpertKey("r1", 0, expert, "layout-v1") for expert in range(3)]
        for key in keys[:2]:
            ticket = directory.begin_upload(key)
            finished = Future(); finished.set_result(None)
            directory.submit(ticket, "weights", b"data",
                lambda t, c, owner, f=finished: m.BundleReceipt(t, c.name, c.encoding, c.nbytes, f))
            directory.publish(ticket)
        return directory, keys

    def test_prefill_never_populates_and_decode_uses_conservative_capacity(self):
        self.assertEqual(p.cache_admission("prefill", 1, 2, 8, 8).mode, "reuse")
        self.assertEqual(p.cache_admission("prefill", 100, 2, 8, 8).mode, "reuse")
        self.assertEqual(p.cache_admission("decode", 2, 2, 8, 4).mode, "populate")
        self.assertEqual(p.cache_admission("decode", 3, 2, 8, 4).mode, "reuse")
        self.assertEqual(p.cache_admission("decode", 33, 2, 8, 8).mode, "reuse")
        self.assertEqual(p.cache_admission("decode", 1, 2, 8, 0).mode, "bypass")
        self.assertEqual(p.cache_admission("decode", 0, 2, 8, 8).mode, "bypass")
        for args in (("score", 1, 2, 8, 8), ("decode", -1, 2, 8, 8),
                     ("decode", 1, 9, 8, 8), ("decode", True, 2, 8, 8)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                p.cache_admission(*args)

    def test_read_only_hit_keeps_replacement_age_but_protects_physical_use(self):
        directory, keys = self.directory()
        ages = tuple(slot.tick for slot in directory.slots)
        first = directory.acquire(keys[0], touch=False)
        second = directory.acquire(keys[1], touch=False)
        self.assertEqual(tuple(slot.tick for slot in directory.slots), ages)
        with self.assertRaisesRegex(ValueError, "protected"):
            directory.begin_upload(keys[2])
        finished = Future()
        with self.assertRaisesRegex(ValueError, "physically settled"):
            directory.settle(first, finished)
        finished.set_result(None)
        directory.settle(first, finished); directory.settle(second, finished)
        ticket = directory.begin_upload(keys[2])
        self.assertEqual(ticket.slot, first.slot)  # prefill did not make this oldest slot hot
        directory.abort(ticket); directory.close()

    def test_populate_hit_still_updates_replacement_age(self):
        directory, keys = self.directory()
        first = directory.acquire(keys[0])
        finished = Future(); finished.set_result(None)
        directory.settle(first, finished)
        ticket = directory.begin_upload(keys[2])
        self.assertNotEqual(ticket.slot, first.slot)
        directory.abort(ticket); directory.close()

    def test_banks_share_only_complete_compatible_layouts_and_revisions(self):
        component = self.component()
        layers = (p.LayerExperts("r1", 0, 8, (component,)), p.LayerExperts("r1", 1, 8, (component,)),
                  p.LayerExperts("r1", 2, 8, (replace(component, shape=(2, 2)),)),
                  p.LayerExperts("r2", 0, 8, (component,)),
                  p.LayerExperts("r1", 3, 8, (replace(component, encoding="demo-i8-layout-v2"),)),
                  p.LayerExperts("r1", 4, 8, (replace(component, expert_bytes=8),)))
        plan = p.size_banks(layers, 4096)
        self.assertEqual(plan.banks[0].layers, (0, 1))
        self.assertEqual(len(plan.banks), 5)
        self.assertEqual(plan.banks[0].slots, 16)
        self.assertLessEqual(plan.nbytes, 4096)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            p.size_banks(layers + (layers[0],), 4096)

    def test_bank_budget_includes_interbank_alignment_and_initialized_guard(self):
        components = (self.component(expert_bytes=3, alignment=4, guard_bytes=3),
                      self.component(name="scales", expert_bytes=2, alignment=16))
        layers = (p.LayerExperts("r1", 0, 8, components, minimum_slots=2),)
        bank = p.size_banks(layers, 32).banks[0]
        self.assertEqual(bank.slots, 4)  # 15 bytes -> 16, then 8 bytes -> 16
        self.assertEqual([(span.offset, span.nbytes) for span in bank.spans], [(0, 16), (16, 16)])
        small = p.size_banks(layers, 24)
        self.assertEqual(small.banks[0].slots, 0)
        self.assertEqual(small.nbytes, 0)
        for budget in range(100):
            plan = p.size_banks(layers, budget)
            self.assertLessEqual(plan.nbytes, budget)
            self.assertEqual(plan.unspent_bytes + plan.nbytes, budget)
            for span in plan.banks[0].spans:
                self.assertEqual(span.offset % span.component.alignment, 0)
                self.assertGreaterEqual(span.nbytes, plan.banks[0].slots*span.component.expert_bytes + span.component.guard_bytes)
        self.assertEqual(p.size_banks((), 32).unspent_bytes, 32)

    def test_bank_geometry_rejects_invalid_components_and_minimum(self):
        for component in (self.component(alignment=0), self.component(guard_bytes=-1),
                          self.component(shape=()), self.component(expert_bytes=True)):
            with self.subTest(component=component), self.assertRaises(ValueError):
                p.size_banks((p.LayerExperts("r1", 0, 8, (component,)),), 32)
        with self.assertRaises(ValueError):
            p.size_banks((p.LayerExperts("r1", 0, 8, (self.component(),), minimum_slots=9),), 32)

    def test_uploads_merge_only_dual_contiguous_ranges_and_preserve_every_pair(self):
        pairs = [(7, 4), (2, 1), (1, 0), (4, 2), (5, 5)]
        runs = p.copy_runs(pairs, self.component(), expert_count=8, slots=6)
        expanded = [(r.first_expert+i, r.first_slot+i) for r in runs for i in range(r.count)]
        self.assertEqual(expanded, sorted(pairs))
        self.assertEqual(len(runs), 4)
        self.assertEqual((runs[0].source_offset, runs[0].destination_offset, runs[0].nbytes), (4, 0, 8))
        source = bytes(range(32)); destination = bytearray(24)
        for run in runs:
            destination[run.destination_offset:run.destination_offset+run.nbytes] = source[run.source_offset:run.source_offset+run.nbytes]
        for expert, slot in pairs:
            self.assertEqual(destination[slot*4:(slot+1)*4], source[expert*4:(expert+1)*4])
        self.assertEqual(destination[12:16], bytes(4))  # gap was not over-copied

    def test_uploads_reject_overwrite_aliases_and_out_of_range_sources(self):
        for pairs in ([(0, 0), (1, 0)], [(0, 0), (0, 1)], [(8, 0)], [(0, 2)], [(True, 0)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                p.copy_runs(pairs, self.component(), expert_count=8, slots=2)

    def test_coalesced_completions_cannot_publish_partial_or_failed_bundles(self):
        directory = m.BundleDirectory(2, (m.BundleComponent("weights", "demo", 4),
                                         m.BundleComponent("scales", "demo-scales", 4)), (object(),))
        tickets = [directory.begin_upload(m.ExpertKey("r1", 0, expert, "layout-v1"))
                   for expert in (1, 2)]
        weights, scales = Future(), Future()
        for component, completion in (("weights", weights), ("scales", scales)):
            for ticket in tickets:
                directory.submit(ticket, component, object(),
                    lambda t, c, owner, f=completion: m.BundleReceipt(t, c.name, c.encoding, c.nbytes, f))
        weights.set_result(None)
        for ticket in tickets:
            with self.assertRaisesRegex(ValueError, "physically settle"):
                directory.publish(ticket)
        self.assertFalse(directory.ready_slots())
        scales.set_exception(OSError("synthetic coalesced scale upload failed"))
        for ticket in tickets:
            with self.assertRaises(OSError):
                directory.publish(ticket)
        self.assertFalse(directory.ready_slots())
        directory.close()

    def test_observation_separates_route_occurrences_from_materialization_demands(self):
        admission = p.cache_admission("decode", 2, 2, 8, 4)
        observed = p.observe_cache(admission, (1, 2, 1, 3), (1,), uploaded_bytes=16,
                                   transfer_calls=2, synchronization_ns=73)
        self.assertEqual((observed.route_hits, observed.route_misses), (2, 2))
        self.assertEqual((observed.unique_hits, observed.unique_misses), (1, 2))
        self.assertEqual((observed.phase, observed.mode, observed.uploaded_bytes,
                          observed.transfer_calls, observed.synchronization_ns), ("decode", "populate", 16, 2, 73))
        with self.assertRaises(ValueError):
            p.observe_cache(admission, (1, 2), (7,))
        with self.assertRaises(ValueError):
            p.observe_cache(admission, (1,), (1,), uploaded_bytes=-1)


if __name__ == "__main__":
    unittest.main()
